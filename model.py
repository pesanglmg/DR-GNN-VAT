# ------------------------------------------------------------------------------
#  DR-GNN + VAT (Robust Graph Recommendation)
#  Modified from original DR-GNN repository by Wenyuer and collaborators.
#  Added Virtual Adversarial Training (VAT) and data poisoning experiments.
#  Reference: https://github.com/WANGBohaO-jpg/DR-GNN.git 
# ------------------------------------------------------------------------------


import copy
import math
import os
import pdb
from matplotlib import pyplot as plt

from tqdm import tqdm
import world
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from world import cprint
DEVICE = torch.device("cpu")

class BasicModel(nn.Module):
    def __init__(self, config: dict, dataset):
        super(BasicModel, self).__init__()
        self.config = config
        self.dataset = dataset

    def getUsersRating(self, users):
        raise NotImplementedError

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g


class LightGCN(BasicModel):
    def __init__(self, config: dict, dataset):
        super(LightGCN, self).__init__(config, dataset)
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["n_layers"]
        self.keep_prob = self.config["keep_prob"]
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config["pretrain"] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            cprint("use NORMAL distribution initilizer")
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config["user_emb"]))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config["item_emb"]))
            print("use pretarined data")
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.Graph
        self.graphdeg = self.dataset.graphdeg
        print(f"lgn is already to go(dropout:{self.config['enable_dropout']})")

    def computer(self, epoch=None, batch_i=None):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        if (self.config["enable_DRO"] and self.training) or (
            self.config["aug_on"] and self.training and epoch >= self.config["aug_warm_up"]
        ):
            L_user, L_item = self.computeL(F.normalize(all_emb, p=2, dim=1), epoch=epoch)
            L = torch.cat([L_user, L_item])

            if self.config["aug_on"] and self.training and epoch >= self.config["aug_warm_up"]:
                # use augmented graph if present, else fall back to normal graph
                temp_graph = getattr(self.dataset, "aug_Graph", self.Graph)
            else:
                temp_graph = self.Graph
            vals = temp_graph.values()
            if L.numel() != vals.numel():
                # fallback: uniform weights for the augmented edges
                L = torch.ones_like(vals, device=vals.device)
                
        else:
            temp_graph = self.Graph
            L = torch.ones_like(self.Graph.values(), device=DEVICE).float()

        g_droped = torch.sparse_coo_tensor(
            temp_graph.indices(), temp_graph.values() * L, temp_graph.size(), device=DEVICE
        ).coalesce()


        if self.config["enable_dropout"] and self.training:
            print("droping")
            g_droped = self.__dropout_x(g_droped, self.keep_prob)

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        if self.config["norm_emb"]:
            users = F.normalize(users, p=2, dim=1)
            items = F.normalize(items, p=2, dim=1)

        return users, items

    def computeL(self, all_emb, epoch, batch_i=None):
        ENABLE_AUG_EDGE = self.config["aug_on"] and (epoch >= self.config["aug_warm_up"])

        if ENABLE_AUG_EDGE:
            if not self.config["full_batch"] and (
                (epoch % self.config["aug_gap"] == 0 and batch_i == 0) or not hasattr(self.dataset, "aug_res_edges")
            ):
                self.dataset.aug_edges(self.embedding_user, self.embedding_item, ratio=self.config["aug_ratio"])
            elif self.config["full_batch"] and (
                epoch % self.config["aug_gap"] == 0 or not hasattr(self.dataset, "aug_res_edges")
            ):
                self.dataset.aug_edges(self.embedding_user, self.embedding_item, ratio=self.config["aug_ratio"])
            edges, aug_edge_coe, aug_edge_coe2 = (
                self.dataset.aug_res_edges.to(DEVICE, non_blocking=True),
                self.dataset.aug_edge_coe.to(DEVICE, non_blocking=True),
                self.dataset.aug_edge_coe2.to(DEVICE, non_blocking=True),
            )

            deg1 = self.graphdeg
        else:
            edges = self.dataset.edges()
            deg1 = self.graphdeg

        U = edges[0]
        I_orig = edges[1]
        I = edges[1] + self.num_users

        if self.config["enable_DRO"]:
            f0 = -(all_emb[U] * all_emb[I]).sum(dim=1) / torch.sqrt(deg1[U] * deg1[I])
            f0 = f0.div(self.config["alpha"])
            f0 = torch.exp(f0).detach()

            if ENABLE_AUG_EDGE:
                f0 = f0 * aug_edge_coe
                f_aug_temp = f0 * aug_edge_coe2

            f_user, f_item = f0, f0
        else:
            if ENABLE_AUG_EDGE:
                f0 = torch.ones_like(U, dtype=torch.float32, device=DEVICE)
                f0 = f0 * aug_edge_coe
                f_aug_temp = f0 * aug_edge_coe2
                f_user, f_item = f0, f0
            else:
                f0 = torch.ones_like(U, dtype=torch.float32, device=DEVICE)
                f_user, f_item = f0, f0

        def cal_Ef(target, f):
            if target == "user":
                num = self.num_users
                idx = U
                deg = deg1[: self.num_users]
            elif target == "item":
                idx = I_orig
                deg = deg1[self.num_users :]
                num = self.num_items
            Ef = torch.zeros(num, device=DEVICE)
            idx = idx.to(torch.long).view(-1).contiguous()   
            f = f.to(Ef.dtype).view(-1).contiguous()
            Ef.scatter_add_(dim=0, index=idx, src=f)
            Ef = Ef / deg
            Ef = torch.where(Ef == 0, torch.tensor(1.0, device=DEVICE), Ef)

            return Ef

        if ENABLE_AUG_EDGE:
            Ef_user = cal_Ef(target="user", f=f_aug_temp)
            L_user = f_user / Ef_user[U]
        else:
            Ef_user = cal_Ef(target="user", f=f_user)
            L_user = f_user / Ef_user[U]
        L_item = L_user

        def resort_L(L, X, Y, target):
            if target == "user":
                col_len = self.num_items
            elif target == "item":
                col_len = self.num_users
            one_dim_indices = X * col_len + Y
            sorted_indices = torch.argsort(one_dim_indices)
            sorted_L = L[sorted_indices]
            return sorted_L

        L_user, L_item = resort_L(L_user, U, I_orig, "user"), resort_L(L_item, I_orig, U, "item")
        return L_user, L_item

    def getUsersRating(self, users, items_tensor=None):
        if items_tensor is None:
            all_users, all_items = self.computer()
            users_emb = all_users[users.long()]
            items_emb = all_items
            rating = self.f(torch.matmul(users_emb, items_emb.t()))
            return rating
        else:
            all_users, all_items = self.computer()
            rating_list = []
            for i, items in enumerate(items_tensor):
                user = users[i].long()
                items = torch.tensor(items, dtype=torch.long, device=DEVICE)

                users_emb = all_users[user].unsqueeze(0)
                items_emb = all_items[items]

                rating_list.append(self.f(torch.matmul(users_emb, items_emb.t())))
            return rating_list

    def getEmbedding(self, users, pos_items, neg_users, neg_items_list, epoch=None, batch_i=None):
        neg_items = neg_items_list.view(-1)

        all_users, all_items = self.computer(epoch, batch_i)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_users_emb = all_users[neg_users]
        neg_emb = all_items[neg_items]

        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)

        return users_emb, pos_emb, neg_users_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    ## ********************* Changes made below FOR VAT integration *********************

    def _raw_logits(self, users_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb_3d: torch.Tensor) -> torch.Tensor:
        """
        Build per-user logits over [pos, neg1..negK].
        users_emb: [B, D]
        pos_emb:   [B, D]
        neg_emb_3d:[B, K, D]
        Returns:   [B, 1+K]
        """
        B, D = users_emb.shape
        K = neg_emb_3d.size(1)
        # [B, 1]
        pos_scores = (users_emb * pos_emb).sum(dim=1, keepdim=True)
        # [B, K]
        neg_scores = (users_emb.unsqueeze(1) * neg_emb_3d).sum(dim=2)
        logits = torch.cat([pos_scores, neg_scores], dim=1)
        # optional temperature for VAT
        temp = float(self.config.get("vat_temp", 1.0))
        if temp != 1.0:
            logits = logits / temp
        return logits

    def _vat_kl(self, p_logit: torch.Tensor, q_logit: torch.Tensor) -> torch.Tensor:
        """
        KL(p || q) where p is fixed (detached), q is current.
        """
        p = torch.softmax(p_logit.detach(), dim=1)
        log_q = torch.log_softmax(q_logit, dim=1)
        return torch.nn.functional.kl_div(log_q, p, reduction="batchmean")

    def _vat_loss(self, users_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb_3d: torch.Tensor) -> torch.Tensor:
        """
        Virtual Adversarial Training on user embeddings (E-VAT).
        Keeps the model locally smooth around users_emb.
        """
        if not self.config.get("enable_vat", False):
            return torch.tensor(0.0)

        xi   = float(self.config.get("vat_xi", 1e-6))
        eps  = float(self.config.get("vat_eps", 2.5))
        n_it = int(self.config.get("vat_ip", 1))
        lam  = float(self.config.get("vat_coeff", 1.0))

        # baseline logits (no grad on the target distribution)
        base_logits = self._raw_logits(users_emb, pos_emb, neg_emb_3d)

        # init small random direction
        d = torch.randn_like(users_emb)
        d = torch.nn.functional.normalize(d, dim=1)
        d = d * xi
        d.requires_grad_(True)

        # power iteration(s)
        for _ in range(n_it):
            y_hat = self._raw_logits(users_emb + d, pos_emb, neg_emb_3d)
            kl = self._vat_kl(base_logits, y_hat)  # KL(p || q(x+d))
            g = torch.autograd.grad(kl, d, retain_graph=True, create_graph=True)[0]
            # update direction
            d = torch.nn.functional.normalize(g.detach(), dim=1) * xi
            d.requires_grad_(True)

        # final adversarial direction
        r_adv = torch.nn.functional.normalize(d.detach(), dim=1) * eps
        y_hat_adv = self._raw_logits(users_emb + r_adv, pos_emb, neg_emb_3d)
        vat_loss = self._vat_kl(base_logits, y_hat_adv)  # KL(p || q(x + r_adv))
        return lam * vat_loss
## ********************* Changes made above FOR VAT integration *********************

    def bprloss(self, users, pos, neg, epoch: int = None):
        neg_num = neg.shape[1]
        neg_users = users.view(-1, 1).repeat(1, neg_num).view(-1)

        (users_emb, pos_emb, neg_users_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(
            users.long(), pos.long(), neg_users.long(), neg.long(), epoch
        )
        reg_loss = (
            (1 / 2)
            * (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2) / neg_num)
            / float(len(users))
        )
        pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(neg_users_emb, neg_emb).sum(dim=1)
        bpr = torch.mean(torch.nn.functional.softplus((neg_scores - pos_scores).div(self.config["tau"])))
        loss = bpr + reg_loss * float(self.config.get("weight_decay", 0.0)) 

       # ------------------- VAT regularization (CPU-safe) -------------------
        if self.config.get("enable_vat", False) and self.training:
            vat_eps   = float(self.config.get("vat_eps",   4.0))   # radius
            vat_xi    = float(self.config.get("vat_xi",    1e-6))  # tiny step for power-iter
            vat_ip    = int(self.config.get("vat_ip",      1))     # power-iter steps
            vat_temp  = float(self.config.get("vat_temp",  1.0))   # temperature on logits
            vat_coeff = float(self.config.get("vat_coeff", 5.0))   # lambda for VAT term

            # two-class logits (pos vs. neg) for each training triple
            def two_class_logits(ps, ns, T=1.0):
                return torch.stack([ps, ns], dim=1) / T

            # p(y|x): original (clean) logits
            with torch.no_grad():
                p_clean = two_class_logits(pos_scores, neg_scores, vat_temp)

            # initialize tiny noise on user/pos/neg embeddings
            ru = torch.randn_like(users_emb)
            rp = torch.randn_like(pos_emb)
            rn = torch.randn_like(neg_emb)
            ru = vat_xi * torch.nn.functional.normalize(ru, dim=1)
            rp = vat_xi * torch.nn.functional.normalize(rp, dim=1)
            rn = vat_xi * torch.nn.functional.normalize(rn, dim=1)

            # power iteration to approximate adversarial direction
            for _ in range(vat_ip):
                ru.requires_grad_()
                rp.requires_grad_()
                rn.requires_grad_()

                ps = ((users_emb + ru) * (pos_emb + rp)).sum(dim=1)
                ns = ((users_emb + ru) * (neg_emb + rn)).sum(dim=1)
                q_noisy = two_class_logits(ps, ns, vat_temp)

                # KL(p||q) with log-target to be numerically stable
                kl = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(p_clean, dim=1),
                    torch.nn.functional.log_softmax(q_noisy, dim=1),
                    log_target=True, reduction="batchmean"
                )

                gu, gp, gn = torch.autograd.grad(kl, [ru, rp, rn], retain_graph=True)
                ru = vat_xi * torch.nn.functional.normalize(gu.detach(), dim=1)
                rp = vat_xi * torch.nn.functional.normalize(gp.detach(), dim=1)
                rn = vat_xi * torch.nn.functional.normalize(gn.detach(), dim=1)

            # final adversarial perturbation with radius eps
            ru = vat_eps * torch.nn.functional.normalize(ru, dim=1)
            rp = vat_eps * torch.nn.functional.normalize(rp, dim=1)
            rn = vat_eps * torch.nn.functional.normalize(rn, dim=1)

            # logits under adversarial perturbation
            ps_adv = ((users_emb + ru) * (pos_emb + rp)).sum(dim=1)
            ns_adv = ((users_emb + ru) * (neg_emb + rn)).sum(dim=1)
            q_adv = two_class_logits(ps_adv, ns_adv, vat_temp)

            vat_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(p_clean, dim=1),
                torch.nn.functional.log_softmax(q_adv,   dim=1),
                log_target=True, reduction="batchmean"
            )

            # add VAT to the training objective
            loss = loss + vat_coeff * vat_loss
        # ---------------------------------------------------------------------

        return loss, reg_loss

    def softmaxloss(self, users, pos, neg, epoch: int = None, batch_i: int = None):
        temp = self.config["ssm_temp"]
        neg_num = neg.shape[1]
        neg_users = users.view(-1, 1).repeat(1, neg_num).view(-1)

        (users_emb, pos_emb, neg_users_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(
            users.long(), pos.long(), neg_users.long(), neg.long(), epoch, batch_i
        )
        reg_loss = (
            (1 / 2)
            * (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2) / neg_num)
            / float(len(users))
        )

        pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(neg_users_emb, neg_emb).sum(dim=1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores.view(-1, neg_num)], dim=1)

        pos_logits = torch.exp(y_pred[:, 0] / temp)
        neg_logits = torch.exp(y_pred[:, 1:] / temp)
        neg_logits = torch.mean(neg_logits, dim=-1)

        loss = -torch.log(pos_logits / neg_logits).mean()

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        mul = torch.sum(inner_pro, dim=1)
        return mul


class MF(BasicModel):
    def __init__(self, config: dict, dataset):
        super(MF, self).__init__(config, dataset)
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config["latent_dim_rec"]
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        cprint("use NORMAL distribution initilizer")
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.Graph
        self.graphdeg = self.dataset.graphdeg
        print(f"mf is already to go")

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        users, items = torch.split(all_emb, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bprloss(self, users, pos, neg, epoch: int = None):
        neg = neg.squeeze()

        (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(
            users.long(), pos.long(), neg.long()
        )
        reg_loss = (
            (1 / 2) * (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / float(len(users))
        )
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus((neg_scores - pos_scores).div(self.config["tau"])))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        mul = torch.sum(inner_pro, dim=1)
        return mul
