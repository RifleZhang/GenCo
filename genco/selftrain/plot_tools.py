import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

COLOR = ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
def plot_embeddings(label_embeddings, embeddings, labels, title=None, figsize=(10, 10), dpi=100):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(np.concatenate([label_embeddings, embeddings]))
    plt.figure(figsize=figsize, dpi=dpi)

    lx, ly = X_tsne[:len(label_embeddings)].T
    px, py = X_tsne[len(label_embeddings):].T
    plt.scatter(px, py, c=labels, alpha=0.8)
    plt.scatter(lx, ly, c=np.arange(len(label_embeddings)), s=100, alpha=1)
    if title:
        plt.title(title)
    plt.legend()
    plt.show()

def plot_embeddings2(embedding_list, label_list, params_list, classes=None, title=None, figsize=(10, 10), dpi=100):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(np.concatenate(embedding_list))
    plt.figure(figsize=figsize, dpi=dpi)
    #fig, ax = plt.subplots()
    s = 0
    for i, labels in enumerate(label_list):
        e = s + len(labels)
        pd = X_tsne[s:e]
        scatter = plt.scatter(pd[:, 0], pd[:, 1], c=labels, cmap=COLOR, **params_list[i])
        handles = scatter.legend_elements()[0]
        s = e
    if title:
        plt.title(title)
    if classes is not None:
        plt.legend(handles=handles, labels=classes)
    plt.show()


# num_train = len(all_train_emb)
# for i in range(dataloader.label_num):
#     ls = num_train + i*2
#     le = ls + 2
#     plt.scatter(X_tsne[ls:le ,0], X_tsne[ls:le,1], c=color[i], s=5, alpha=1, label=i)
#     idx = train_labels == i
#     plt.scatter(X_tsne[:num_train][idx, 0], X_tsne[:num_train][idx, 1], c=color[i], s=0.5, alpha=0.5)
# plt.legend()
# plt.show()