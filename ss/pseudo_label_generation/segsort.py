import torch
from torch.nn import functional as F


def calculate_prototypes_from_labels(embedding, labels):
    """
    Calculates prototypes from labels.
    This function calculates prototypes (mean direction) from embedding features_ours.pickle for each label.
    This function is also used as the m-step in k-means
    clustering.
    Args:
        embedding: embedding, with shape [batch_size, embedding_dim, height, weight]
        labels: An N-D int32 label map for each embedding pixel.
            with shape [batch_size, 1, height, width]
    Returns:
        prototypes: A 2-D float tensor with shape `[num_prototypes, embedding_dim]`.
    """
    # reshape embedding to a 2-D tensors
    embedding = embedding.permute(0, 2, 3, 1)  # (B, H, W, C)
    embedding = embedding.reshape(-1, embedding.shape[-1])  # (B * H * W, C)
    # convert label to one-hot encoding
    labels = labels.reshape(-1).long()
    one_hot_labels = F.one_hot(labels, num_classes=-1).float()
    # calculate prototypes
    one_hot_labels = one_hot_labels.permute(1, 0)  # (max_label, B * H * W)
    prototypes = torch.matmul(one_hot_labels, embedding)  # (max_label, C)
    # normalise prototype
    norm = torch.norm(prototypes, dim=1, keepdim=True)
    if 0 in norm:
        # when not called from add_unsupervised_segsort_loss, there may be cluster label with zero pixels
        norm[norm == 0] = 1
    prototypes = prototypes / norm  # (max_label, C)
    return prototypes


def _calculate_similarities(embedding, prototypes, concentration, split=1):
    """Calculates cosine similarities between embedding and prototypes.
    This function calculates cosine similarities between embedding and prototypes.
    It splits the matrix multiplication to prevent an unknown bug in Tensorflow.
    Args:
        embedding: [batch_size, embedding_dim, height, width)
        prototypes: A 2-D float tensor with shape [num_prototypes, embedding_dim].
        concentration: A float that controls the sharpness of cosine similarities.
        split: An integer for number of splits of matrix multiplication.
    Returns:
        similarities: A 2-D float tensor with shape [num_pixesl, num_prototypes].
    """
    embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding.shape[1])
    prototypes = prototypes.permute(1, 0)  # (embedding_dim, num_prototype)
    if split > 1:
        step_size = embedding.shape[0] // split
        for s in range(split):
            if s < split - 1:
                embedding_temp = embedding[step_size*s:step_size*(s+1)]
            else:
                embedding_temp = embedding[step_size*s:]
            # both embedding and prototype have been normalised, therefore no need to divide by norm
            pre_similarities_temp = torch.matmul(embedding_temp, prototypes) * concentration
            # (num_pixel, num_prototypes)
            if s == 0:
                pre_similarities = pre_similarities_temp
            else:
                pre_similarities = torch.cat([pre_similarities, pre_similarities_temp], 0)  # (num_pixel, nun_prototype)
    else:
        pre_similarities = torch.matmul(embedding, prototypes) * concentration
    similarities = torch.exp(pre_similarities)
    return similarities


def get_cluster_loss(embedding,
                     concentration,
                     cluster_labels,):
    """
    calculate cluster loss
    Args:
        embedding:  [batch_size, embedding_dim, height, weight]
        concentration: A float that controls the sharpness of cosine similarities.
        cluster_labels:A 4-D integer tensor of a contour segmentation
            mask with shape [batch_size, 1, height, width]
    Returns:
        loss: torch.tensor(float)
    """
    # resize cluster_labels to match embedding shape
    cluster_labels = F.interpolate(cluster_labels.float(), size=embedding.shape[-2:])

    # normalize embedding.
    batch_size,  embedding_dim, _, _ = embedding.shape
    embedding = embedding / torch.norm(embedding, dim=1, keepdim=True)  # (B, C, H, W)

    # add offset to cluster labels, such that all batches could be calculated in parallel
    max_clusters = torch.max(cluster_labels)
    offset = torch.arange(0, max_clusters * batch_size, max_clusters, device=cluster_labels.device)  # (B)
    cluster_labels += offset.reshape(-1, 1, 1, 1)  # (B, 1, H, W)
    _, cluster_labels = torch.unique(cluster_labels.reshape(-1),
                                     sorted=False,
                                     return_inverse=True)   # (B, 1, H, W)

    # calculate the similarity matrix representing similarities between all pixel-cluster pairs
    prototypes = calculate_prototypes_from_labels(embedding, cluster_labels)  # (max_label, C)
    similarities = _calculate_similarities(
        embedding, prototypes, concentration, batch_size)  # (num_pixels, num_prototypes)

    # calculate the negative log likelihood of all pixels are assigned to the right cluster
    num_pixel, num_prototype = similarities.shape
    cluster_labels = cluster_labels.reshape(-1)  # (B * H * W)
    indices = torch.tensor([i * num_prototype + cluster_labels[i] for i in range(num_pixel)],
                           device=similarities.device)
    numerator = similarities.take(indices)
    denominator = torch.sum(similarities, dim=1)  # (num_pixels)
    probabilities = numerator / denominator  # (num_pixels)
    return torch.mean(- torch.log(probabilities))


def add_location_features_and_normalise(embedding):
    """Add location features_ours.pickle to embedding and normalise
    Args:
        embedding: [batch_size, embedding_dim, height, width]
    Returns:
        embedding_with_location: [batch_size, embedding_dim + 2, height, width]
    """
    # initialise location features_ours.pickle
    batch_size, embedding_dim, height, width = embedding.shape
    img_dimensions = torch.tensor([height, width])
    y_features = torch.arange(img_dimensions[0]).unsqueeze(1).expand(-1, img_dimensions[1])
    x_features = torch.arange(img_dimensions[1]).unsqueeze(0).expand(img_dimensions[0], -1)
    location_features = torch.stack([x_features, y_features], dim=0)
    location_features = location_features.cuda().float()
    location_features = location_features.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, 2, H, W)
    # concatenate location_features to embedding
    embedding_with_location = torch.cat([embedding, location_features], 1)  # (B, C, H, W)
    # normalise
    embedding_with_location = embedding_with_location / torch.norm(embedding_with_location, dim=1, keepdim=True)
    return embedding_with_location


def initialize_cluster_labels(num_clusters, img_dimensions):
    """Initializes uniform cluster labels for an image.
    This function is used to initialize cluster labels that uniformly partition
    a 2-D image.
    Args:
        num_clusters: A list of 2 integers for number of clusters in y and x axes.
        img_dimensions: A list of 2 integers for image's y and x dimension.
    Returns:
        A 2-D int32 tensor with shape specified by img_dimension.
    """
    yx_range = torch.ceil(img_dimensions.float() / num_clusters.float()).int()  # (2)
    y_labels = torch.tensor([y / yx_range[0] for y in range(img_dimensions[0])]).unsqueeze(1)  # (img_dim[1], 1)
    x_labels = torch.tensor([x / yx_range[1] for x in range(img_dimensions[1])]).unsqueeze(0)  # (1, img_dim[1])
    labels = y_labels + (torch.max(y_labels) + 1) * x_labels  # (num_clusters[0], num_clusters[1])
    labels = labels.cuda()
    return labels


def find_nearest_prototypes(embedding, prototypes):
    """Finds the nearest prototype for each embedding pixel.
    This function calculates the index of nearest prototype for each embedding
    pixel. This function is also used as the e-step in k-means clustering.
    Args:
        embedding: An 3-D float tensor with shape [embedding_dim, height, weight].
        prototypes: A 2-D float tensor with shape `[num_prototypes, embedding_dim]`.
    Returns:
        A 3-D int32 tensor with shape [1, height, width] containing the index of the
            nearest prototype for each pixel.
    """
    # reshape embedding
    embedding_dim, height, width = embedding.shape
    embedding = embedding.permute(1, 2, 0).reshape(-1, embedding_dim)  # (H * W, C)
    # calculate the similarity matrix representing similarities between all pixel-cluster pairs
    similarities = torch.matmul(embedding, prototypes.permute(1, 0))  # (H * W, num_prototype)
    # choose the prototype with highest similarity
    nearest_prototype = torch.argmax(similarities, dim=1)  # (H * W)
    # reshape output back to original size
    nearest_prototype = nearest_prototype.reshape(1, height, width)
    return nearest_prototype


def kmeans(embedding,
           num_clusters=torch.tensor([5, 5]),
           iterations=10):
    """Performs the von-Mises Fisher k-means clustering with initial labels.
    Args:
        embedding: A 3-D float tensor with shape `[embedding_dim, height, weight]`.
        num_clusters: An integer for the maximum of labels.
        iterations: Number of iterations for the k-means clustering.
    Returns:
        A 3-D integer tensor of the cluster label for each pixel, with shape [1, height, width]
    """
    embedding = embedding.unsqueeze(0)  # (1, C, H, W)
    height, width = embedding.shape[-2:]
    cluster_labels = initialize_cluster_labels(
        num_clusters,
        torch.tensor([height, width]))  # (H, W)
    cluster_labels = cluster_labels.reshape(1, 1, height, width)
    for _ in range(iterations):
        # M-step of the vMF k-means clustering.
        prototypes = calculate_prototypes_from_labels(embedding, cluster_labels)  # (num_prototypes, C)
        # E-step of the vMF k-means clustering.
        cluster_labels = find_nearest_prototypes(embedding[0], prototypes).unsqueeze(0)  # (1, 1, H, W)
    return cluster_labels.squeeze(0)


def get_cluster_labels(embedding,
                       num_clusters=torch.tensor([7, 7]),
                       kmeans_iterations=10,):
    """
    cluster pixels based on the embedding
    Args:
        embedding: A 4-D float tensor with shape `[batch_size, embedding_dim, height, width]`.
        num_clusters: An integer scalar for total number of clusters.
        kmeans_iterations: Number of iterations for the k-means clustering.
    Returns:
        cluster_labels: [batch_size, 1, height, weight]
    """
    embedding_with_location = add_location_features_and_normalise(embedding)
    cluster_labels = []
    for e in embedding_with_location:
        cl = kmeans(
            e,  # (C, H, W)
            num_clusters,
            kmeans_iterations
        )  # (1, H, W)
        cluster_labels.append(cl)
    cluster_labels = torch.stack(cluster_labels, dim=0)  # (B, 1, H, W)
    cluster_labels = F.interpolate(cluster_labels.float(), size=(473, 473))
    return cluster_labels


def get_cluster_labels_superpixel(embedding,
                                  superpixel_labels,
                                  num_clusters=torch.tensor([7, 7]),
                                  kmeans_iterations=10):
    """
    cluster superpixels based on embedding
    Args:
        embedding: [batch_size, embedding_dim, height, width]
        superpixel_labels: [batch_size, 1, height, width]
        num_clusters: int
        kmeans_iterations: int
    Returns:
        cluster_labels: [batch_size, 1, height, width]
    """
    # resize embedding to fit the size of superpixel_labels
    embedding = F.interpolate(embedding,
                              size=superpixel_labels.shape[-2:],
                              mode='bilinear',
                              align_corners=True)
    # add location features_ours.pickle
    embedding_with_location = add_location_features_and_normalise(embedding)
    # do k-means for each batch
    cluster_labels = []
    for e, sl in zip(embedding_with_location, superpixel_labels):  # (C, H, W), (1, H, W)
        u_sl, sl = torch.unique(sl, return_inverse=True)
        # calculate superpixel embeddings
        superpixel_embedding = calculate_prototypes_from_labels(
            e.unsqueeze(0),
            sl.unsqueeze(0))  # (number of superpixels, C)
        superpixel_embedding = superpixel_embedding.permute(1, 0).unsqueeze(-1)  # (C, number of superpixels, 1)
        # do k-means in units of superpixels
        scl = kmeans(
            superpixel_embedding,
            num_clusters,
            kmeans_iterations)[0, :, 0]  # (number of superpixels)
        # index back to pixels
        cl = torch.zeros_like(sl)  # (1, H, W)
        for i in range(len(u_sl)):
            cl[sl == i] = scl[i]
        cluster_labels.append(cl)

    cluster_labels = torch.stack(cluster_labels, dim=0)  # (B, 1, H, W)
    cluster_labels[superpixel_labels == 255] = 255
    return cluster_labels


def superpixel_pool(embedding,
                    superpixel_labels,):
    """
    Args:
        embedding: [batch_size, embedding_dim, height, width]
        superpixel_labels: [batch_size, 1, height, width]
    Returns:
        output: [batch_size, embedding_dim, height, width]
    """
    batch_size = embedding.shape[0]
    # resize embedding to fit the size of superpixel_labels
    embedding = F.interpolate(embedding,
                              size=superpixel_labels.shape[-2:],
                              mode='bilinear',
                              align_corners=True)

    # add offset to cluster labels, such that all batches could be calculated in parallel
    max_clusters = torch.max(superpixel_labels)
    offset = torch.arange(0, max_clusters * batch_size, max_clusters, device=superpixel_labels.device)  # (B)
    superpixel_labels += offset.reshape(-1, 1, 1, 1)  # (B, 1, H, W)
    _, superpixel_labels = torch.unique(superpixel_labels,
                                        sorted=False,
                                        return_inverse=True)  # (B, 1, H, W)

    # calculate the similarity matrix representing similarities between all pixel-cluster pairs
    prototypes = calculate_prototypes_from_labels(embedding, superpixel_labels)  # (max_label, C)
    output = torch.ones_like(embedding).permute(0, 2, 3, 1)  # (B, H, W, embedding_dim)
    for i, p in enumerate(prototypes):
        output[superpixel_labels[:, 0, :, :] == i] = p
    output = output.permute(0, 3, 1, 2)
    return output