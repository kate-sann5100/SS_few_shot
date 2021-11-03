import random

import torch
from torch import nn
from torch.nn import functional as F

from ss.model.backbone.resnet import get_resnet_layer
from ss.model.backbone.vgg_pfenet import get_vgg16_layer
from ss.pseudo_label_generation.segsort import calculate_prototypes_from_labels


def Weighted_GAP(feat, mask):
    """
    Perform global average pooling over masked support features
    :param feat: (B, S, C, h, w)
    :param mask: (B, S, 1, H, W)
    :return:
        feat: (B, S, C, 1, 1)
    """
    if len(feat) != len(mask) or feat.shape[-2] != mask.shape[-2] or feat.shape[-1] != mask.shape[-1]:
        raise RuntimeError(f'expecting same shape input for Weighted_GAP,'
                           f'got feat of shape {feat.shape},'
                           f'mask of shape {mask.shape}')
    h, w = feat.shape[-2:]
    area = torch.sum(mask, dim=(-1, -2), keepdim=True) * h * w + 0.0005  # (B, S, 1, 1, 1)
    feat = torch.sum(feat * mask, dim=(-1, -2), keepdim=True) * h * w / area  # (B, S, C, 1, 1)

    return feat


class PFENet(nn.Module):

    def __init__(self, backbone, self_supervision, prior):
        super(PFENet, self).__init__()

        self.pyramid_bins = [60, 30, 15, 8]
        self.self_supervision = self_supervision
        self.prior = prior

        self.backbone_name = backbone
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4, fea_dim = self.init_backbone(backbone)
        self.backbone_layers = [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]

        self.down_query, self.down_supp = self.init_head(fea_dim)
        self.comparison_layers = [self.down_query, self.down_supp]

        self.init_merge, self.alpha_conv, self.beta_conv, self.inner_cls, self.res1, self.res2, self.cls = self.init_fem()
        self.comparison_layers.extend(
            [self.init_merge, self.alpha_conv, self.beta_conv, self.inner_cls, self.res1, self.res2, self.cls]
        )

        self.avg_pool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avg_pool_list.append(nn.AdaptiveAvgPool2d(bin))

    @staticmethod
    def init_backbone(backbone_name):
        """
        load ImageNet pretrained backbones
        for ResNet, convert stride convolution after layer2 to dilated convolution to maintain a output stride of 8
        Args:
            backbone_name: name of backbone, 'vgg16' or 'resnet50' or 'resnet101'
        Returns:
            initialised layers, and
            feat_dim: dim of concatenated feature from layer2 and layer3, int
        """
        print(f'initialising {backbone_name} backbone')
        if 'vgg' in backbone_name:
            layer0, layer1, layer2, layer3, layer4 = get_vgg16_layer()
            fea_dim = 512 + 256
        else:
            layer0, layer1, layer2, layer3, layer4 = get_resnet_layer(backbone_name)
            fea_dim = 1024 + 512

        return layer0, layer1, layer2, layer3, layer4, fea_dim

    def init_head(self, feat_dim):
        down_query = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        down_supp = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        return down_query, down_supp

    def init_fem(self, reduce_dim=256):
        """
        initialise a feature enrichment module (FEM), which includes
        - a same set of layers (init_merge, alpha_conv, beta_conv, inner_cls) are initialised for each scale
            init_merge merges the query, support, and prior
            alpha_conv extracts useful information from the higher scale features_ours.pickle
            beta_conv finishes the inter-scale interaction and output the refined feature
            inner_cls produces scale-wise prediction to calculate the auxiliary loss for intermediate supervision
        res1 reduce the concatenated output of all scales to reduce_dim
        res2 and cls makes the final prediction
        Args:
            reduce_dim: input dimensions of all layers except for res2, int
            prior: if true, concatenate training-free prior mask to support and query feature, bool
            cluster: if true, FEM generated is used for 32-d cluster feature extraction, bool
        Returns:
            initialised layers
        """
        init_merge = nn.ModuleList()
        alpha_conv = nn.ModuleList()
        beta_conv = nn.ModuleList()
        inner_cls = nn.ModuleList()

        in_dim = reduce_dim * 2
        if self.prior:
            in_dim += 1

        for i, _ in enumerate(self.pyramid_bins):
            init_merge.append(nn.Sequential(
                nn.Conv2d(in_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))
            inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(reduce_dim, 2, kernel_size=1)
            ))
            if i != 0:
                alpha_conv.append(nn.Sequential(
                    nn.Conv2d(2 * reduce_dim, reduce_dim, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.ReLU()
                ))

        res1 = nn.Sequential(
            nn.Conv2d(reduce_dim * len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, 2, kernel_size=1)
        )

        return init_merge, alpha_conv, beta_conv, inner_cls, res1, res2, cls

    def forward_backbone(self, x):
        """
        Args:
            x: rgb image input, (B, C, H, W)
        Returns:
            feat_1: output of layer1
            feat_2: output of layer2
            feat_3: output of layer3
            feat_4: output of layer4
        """
        with torch.no_grad():
            feat_0 = self.layer0(x)
            feat_1 = self.layer1(feat_0)
            feat_2 = self.layer2(feat_1)
            feat_3 = self.layer3(feat_2)
            feat_4 = self.layer4(feat_3)
            if 'vgg' in self.backbone_name:
                feat_2 = F.interpolate(feat_2, size=(feat_3.size(2), feat_3.size(3)),
                                       mode='bilinear', align_corners=True)
        return feat_1, feat_2, feat_3, feat_4

    def forward_comparison(self, query_feat, query_feat_4, supp_feat=None, supp_feat_4=None, supp_y=None, ss=False):
        """
        compare query with support to make pixelwise prediction
        Args:
            query_feat: concatenated layer2+3 feature, (B, C, H, W)
            query_feat_4: layer4 feature (B, C, H, W)
            supp_feat: concatenated layer2+3 feature, (B*S, C, H, W)
            supp_feat_4: layer4 feature (B*S, C, H, W)
            supp_y: support mask (B, S, H, W)
        Returns:
            out: (B, 2, H. W)
            out_list: list of (B, 2, H, W)
        """
        b, s = supp_y.shape[:2]
        h, w = query_feat.shape[-2:]

        query_feat = self.down_query(query_feat)  # (B, C, h, w)
        supp_feat = self.down_supp(supp_feat)  # (B*shot, C, h, w)

        supp_feat = supp_feat.reshape(b, s, -1, h, w)
        supp_feat_4 = supp_feat_4.reshape(b, s, -1, h, w)
        mask = torch.zeros_like(supp_y)  # (B, S, H, W)
        mask[supp_y == 1] = 1
        mask = F.interpolate(mask.float(), (h, w), mode='bilinear', align_corners=True)
        mask = mask.unsqueeze(2)  # (B, S, 1, H, W)

        prototype = Weighted_GAP(supp_feat, mask)  # (B, S, C, 1, 1)
        prototype = torch.mean(prototype, dim=1)  # (B, C, 1, 1)

        if self.prior:
            if ss:
                prior = torch.ones_like(query_feat[:, :1, :, :]) * 0.5
            else:
                masked_supp_feat_4 = supp_feat_4 * mask
                prior = self.get_corr_mask(query_feat_4, masked_supp_feat_4)  # (B, 1, h, w)
        else:
            prior = None
        out, out_list = self.forward_FEM(query_feat, prototype, prior)
        return out, out_list

    @staticmethod
    def get_corr_mask(x, y):
        """
        generate the train-free prior, by the following steps
        - compute a similarity matrix representing correspondence between each x and y pixel pair
        - assign the highest similarity score to each x pixel to generate the prior
        - min-max normalise the x
        Args:
            x: (B, C, h, w)
            y: (B, S, C, h, w)
        Returns:
            prior: train-free prior, (B, 1, h, w)
        """
        b, s, c, h, w = y.shape

        x = x.unsqueeze(1).expand(-1, s, -1, -1, -1)  # (B, S, C, h, w)
        x = x.contiguous().view(b * s, c, h * w)  # (B*S, C, h*w)
        y = y.contiguous().view(b * s, c, h * w)  # (B*S, C, h*w)
        y = y.permute(0, 2, 1)  # (B*S, h*w, C)

        # compute norm
        x_norm = torch.norm(x, 2, 1, True)  # (B*S, 1, h*w)
        y_norm = torch.norm(y, 2, 2, True)  # (B*S, h*w, 1)

        # compute cosine similarity
        cosine_eps = 1e-7
        similarity = torch.bmm(y, x) / (torch.bmm(y_norm, x_norm) + cosine_eps)  # (B*shot, h*w, h*w)

        # choose the highes similarity score for each query pixel
        similarity = similarity.max(1)[0].view(b * s, h * w)  # (B*S, h*w)

        # min-max normalisation
        min = torch.min(similarity, dim=1, keepdim=True)[0]
        max = torch.max(similarity, dim=1, keepdim=True)[0]
        similarity = (similarity - min) / (max - min + cosine_eps)  # (B*shot, h*w)

        # mean reduce over shots
        similarity = similarity.reshape(b, s, h, w)
        prior = torch.mean(similarity, dim=1, keepdim=True)  # (B, 1, h, w)
        return prior

    def forward_FEM(self, query_feat, prototype, prior):
        """
        :param query_feat: (B, C, h, w)
        :param prototype: (B, C, 1, 1)
        :param prior: (B, 1, h, w)
        :return:
            out: (B, 2, h, w)
            out_list: list of (B, 2, h, w)
        """
        h, w = query_feat.shape[-2:]
        out_list = []
        pyramid_feat_list = []
        for idx, tmp_bin in enumerate(self.pyramid_bins):
            # resize query feature to current scale
            bin = tmp_bin
            query_feat_bin = self.avg_pool_list[idx](query_feat)

            # inter-source interaction
            # merge support feature and prior to query feature
            supp_feat_bin = prototype.expand(-1, -1, bin, bin)
            merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin], 1)
            if self.prior:
                corr_mask_bin = F.interpolate(prior, size=(bin, bin), mode='bilinear', align_corners=True)
                merge_feat_bin = torch.cat([merge_feat_bin, corr_mask_bin], 1)
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)

            # inter-scale interaction
            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx - 1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx - 1](rec_feat_bin) + merge_feat_bin
            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(h, w), mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)

        # make final prediction based on information from all scales
        query_feat = torch.cat(pyramid_feat_list, 1)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat
        out = self.cls(query_feat)
        return out, out_list

    def criterion(self, pred, pred_list, target, get_aux_loss=False):
        """
        calculate loss
        Args:
            pred: (B, 2, H, W)
            pred_list: list of (B, 2, h, w)
            target: (B, H, W)
            get_aux_loss: if true calculate aux_loss, otherwise set aux_loss to 0, bool
        """
        main_loss = F.cross_entropy(pred, target.long(), ignore_index=255)
        aux_loss = torch.zeros_like(main_loss)
        if get_aux_loss:
            for idx_k in range(len(pred_list)):
                inner_out = pred_list[idx_k]
                inner_out = F.interpolate(inner_out, size=target.shape[-2:], mode='bilinear', align_corners=True)
                a_l = F.cross_entropy(inner_out, target.long(), ignore_index=255)
                aux_loss = aux_loss + a_l
            aux_loss = aux_loss / len(pred_list)
        return main_loss, aux_loss

    def choose_cluster(self, reference_feat, cluster_label, y):
        """
        choose the cluster used for counter_fg_loss, conditioned on the activation intensity
        Args:
            reference_feat: (B, C, H, W)
            cluster_label: (B, H, W)
            y: (B, H, W)
        Returns:
            mask: (B, H, W)
        """
        batch_size, H, W = cluster_label.shape
        reference_feat = F.interpolate(reference_feat, size=(H, W), mode='bilinear', align_corners=True)
        cluster_label[y == 1] = 0  # (B, H, W)

        mask = []
        # loop over batches
        for f, cl in zip(reference_feat, cluster_label):  # (C, H, W), (H, W)
            # compute prototypes of each cluster
            prototypes = calculate_prototypes_from_labels(f.unsqueeze(0), cl.unsqueeze(0).unsqueeze(0))
            # (num_superpixels, C)
            # choose one cluster from the 10 with highest activations
            activation = torch.sum(prototypes, dim=1)  # (num_superpixels)
            k = min(len(activation), 5)
            top_5 = torch.topk(activation, k=k, dim=0)[1]
            chosen_index = [i for i in top_5 if i != 0 and i != 255]
            if len(chosen_index) == 0:
                mask.append(cl == 0)  # (H, W)
            else:
                mask.append(cl == chosen_index[random.randint(0, len(chosen_index) - 1)])  # (H, W)

        mask = torch.stack(mask, dim=0).long()  # (B, H, W)
        return mask

    def forward(self, input_dict):
        x = input_dict['q_x']  # (B, 3, H, W)
        y = input_dict['q_y']  # (B, H, W)
        s_x = input_dict['s_x']  # (B, S, 3, H, W)
        s_y = input_dict['s_y']  # (B, S, H, W)

        b, shot, _, H, W = s_x.shape
        s_x = s_x.reshape(-1, 3, H, W)  # (B*shot, 3, h, w)

        query_feat_1, query_feat_2, query_feat_3, query_feat_4 = self.forward_backbone(x) # (B, C, h, w)
        query_feat = torch.cat([query_feat_3, query_feat_2], dim=1)  # (B, C, h, w)

        # supervised pathway
        supp_feat_1, supp_feat_2, supp_feat_3, supp_feat_4 = self.forward_backbone(s_x)  # (B*shot, C, h, w)
        supp_feat = torch.cat([supp_feat_3, supp_feat_2], dim=1)  # (B*shot, C, h, w)
        out, out_list = self.forward_comparison(
            query_feat, query_feat_4,
            supp_feat, supp_feat_4, s_y
        )
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        out_binary = out.argmax(dim=1)

        out_dict = dict(out=out, out_binary=out_binary)

        if self.training:
            supervised_target = y.clone()
            supervised_target[y == 2] = 0
            supervised_target[y == 3] = 0
            main_loss, aux_loss = self.criterion(out, out_list, supervised_target, get_aux_loss=self.training)
            out_dict['main_loss'] = main_loss
            out_dict['aux_loss'] = aux_loss
            if self.self_supervision:
                with torch.no_grad():
                    superpixel_label = input_dict['superpixel']  # (B, H, W)
                    pseudo_mask = torch.zeros_like(y)  # (B, H, W)
                    chosen_cluster = self.choose_cluster(
                        reference_feat=query_feat,
                        cluster_label=superpixel_label,
                        y=y)  # (B, H, W)
                    pseudo_mask[chosen_cluster == 1] = 1
                    pseudo_mask[y == 1] = 0  # (B, H, W)

                    # ignore image if counter_fg_area is too small
                    for b_idx, sum in enumerate(torch.sum(pseudo_mask == 1, dim=(1, 2))):
                        if sum < 32 * 32 * 2:
                            pseudo_mask[b_idx, :, :] = 255
                # make predictions
                pseudo_out, pseudo_out_list = self.forward_comparison(
                    query_feat=query_feat,   # (B, C, H, W)
                    query_feat_4=query_feat_4,  # (B, C, H, W)
                    supp_feat=query_feat,  # (B, C, H, W)
                    supp_feat_4=query_feat_4,  # (B, C, H, W)
                    supp_y=pseudo_mask.unsqueeze(1),  # (B, 1, H, W)
                    ss=True)
                # compute loss
                pseudo_out = F.interpolate(pseudo_out, size=(H, W), mode='bilinear', align_corners=True)
                self_supervised_main_loss, self_supervised_fg_aux_loss = self.criterion(
                    pred=pseudo_out,  # (B, 2, H. W)
                    pred_list=pseudo_out_list,  # list of (B, 2, h. w)
                    target=pseudo_mask,  # (B, H. W)
                    get_aux_loss=True)
                out_dict['self_supervised_loss'] = self_supervised_main_loss + self_supervised_fg_aux_loss
            else:
                out_dict['self_supervised_loss'] = torch.zeros_like(main_loss)

        return out_dict