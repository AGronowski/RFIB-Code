import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

<<<<<<< HEAD
# Calculate Renyi divergence between two multivariate Gaussians
# mu is mean of 1st distribution, mean of 2nd distribution is 0
# var is variance of 1st distribution, gamma is variance of 2nd distribution
def renyi_divergence(mu, var, alpha=5, gamma=1):
    sigma_star = alpha * gamma + (1 - alpha) * var
    term1 = alpha / 2 * mu ** 2 / sigma_star

    term2_1 = var ** (1 - alpha) * gamma ** alpha
=======

def renyi_divergence(mu, log_var, alpha):
    var = log_var.exp()

    sigma_star = alpha + (1 - alpha) * var
    term1 = alpha / 2 * mu.pow(2) * sigma_star

    term2_1 = var.pow(1 - alpha)
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
    term2_2 = torch.log(sigma_star / term2_1)
    term2 = -0.5 / (alpha - 1) * term2_2

    total = term1 + term2

    return torch.sum(total)


<<<<<<< HEAD
# IB or CFB loss
def get_KLdivergence_loss(yhat, y, mu, logvar, beta):
=======
# original loss using kl divergence
def get_IB_or_Skoglund_original_loss(yhat, y, mu, logvar, beta):
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
    divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    yhat = torch.where(torch.isnan(yhat), torch.zeros_like(yhat), yhat)
    yhat = torch.where(torch.isinf(yhat), torch.zeros_like(yhat), yhat)

    cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(yhat.view(-1), y,
                                                                         reduction='sum')
<<<<<<< HEAD
    return divergence + beta * cross_entropy


# RFIB loss
def get_RFIB_loss(yhat, yhat_fair, y, mu, logvar, alpha, beta1, beta2):
=======

    return divergence + beta * cross_entropy


def get_combined_loss(yhat, yhat_fair, y, mu, logvar, alpha, beta1, beta2):
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
    if alpha == 0:
        divergence = 0
    elif alpha == 1:  # KL Divergence
        divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        divergence = renyi_divergence(mu, logvar, alpha)

    IB_cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(yhat.view(-1), y,
                                                                            reduction='sum')
<<<<<<< HEAD
    CFB_cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(yhat_fair.view(-1), y,
                                                                             reduction='sum')

    loss = divergence + beta1 * IB_cross_entropy + beta2 * CFB_cross_entropy
=======
    Skoglund_cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(yhat_fair.view(-1), y,
                                                                                  reduction='sum')

    loss = divergence + beta1 * IB_cross_entropy + beta2 * Skoglund_cross_entropy
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

    return loss
