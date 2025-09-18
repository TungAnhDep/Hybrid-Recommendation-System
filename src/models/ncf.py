import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridRecommender(nn.Module):
    def __init__(self, usr, prd, feat, ctx, embedding_size=20, dropout=0.2):
        super(HybridRecommender, self).__init__()

        # ====== Collaborative Filtering ======
        # A) Matrix Factorization
        self.cf_user_emb = nn.Embedding(usr, embedding_size)
        self.cf_product_emb = nn.Embedding(prd, embedding_size)

        # B) Neural Network part
        self.nn_user_emb = nn.Embedding(usr, embedding_size)
        self.nn_product_emb = nn.Embedding(prd, embedding_size)
        self.nn_dense = nn.Linear(embedding_size * 2, embedding_size // 2)
        self.nn_dropout = nn.Dropout(dropout)

        # ====== Content Based ======
        self.features_dense = nn.Linear(feat, feat)
        self.features_dropout = nn.Dropout(dropout)

        # ====== Knowledge Based ======
        self.context_dense = nn.Linear(ctx, ctx)
        self.context_dropout = nn.Dropout(0.1)

        # ====== Output ======
        self.out_dense = nn.Linear(1 + (embedding_size // 2) + feat + ctx, 1)

    def forward(self, users, products, features, contexts):
        """
        users: LongTensor (batch,)
        products: LongTensor (batch,)
        features: FloatTensor (batch, feat)
        contexts: FloatTensor (batch, ctx)
        """

        # ====== CF - A) Matrix Factorization ======
        cf_u = self.cf_user_emb(users)              # (batch, emb)
        cf_p = self.cf_product_emb(products)        # (batch, emb)
        # cosine similarity (dot product normalized)
        cf_xx = F.cosine_similarity(cf_u, cf_p).unsqueeze(1)  # (batch, 1)

        # ====== CF - B) Neural Network ======
        nn_u = self.nn_user_emb(users)      # (batch, emb)
        nn_p = self.nn_product_emb(products) # (batch, emb)
        nn_x = torch.cat([nn_u, nn_p], dim=1)    # (batch, emb*2)
        nn_x = F.relu(self.nn_dense(nn_x))       # (batch, emb/2)
        nn_x = self.nn_dropout(nn_x)

        # ====== Content Based ======
        feat_x = F.relu(self.features_dense(features))  # (batch, feat)
        feat_x = self.features_dropout(feat_x)

        # ====== Knowledge Based ======
        ctx_x = F.relu(self.context_dense(contexts))    # (batch, ctx)
        ctx_x = self.context_dropout(ctx_x)

        # ====== Merge all ======
        concat = torch.cat([cf_xx, nn_x, feat_x, ctx_x], dim=1)

        y_out = self.out_dense(concat)   # (batch, 1)

        return y_out
