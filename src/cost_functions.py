import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Calculate Renyi divergence between two multivariate Gaussians
# mu is mean of 1st distribution, mean of 2nd distribution is 0
# var is variance of 1st distribution, gamma is variance of 2nd distribution
def renyi_divergence(mu, var, alpha, gamma=1):
    sigma_star = alpha * gamma + (1 - alpha) * var
    term1 = alpha / 2 * mu ** 2 / sigma_star

    term2_1 = var ** (1 - alpha) * gamma ** alpha
    term2_2 = torch.log(sigma_star / term2_1)
    term2 = -0.5 / (alpha - 1) * term2_2

    total = term1 + term2

    return torch.sum(total)

def renyi_crossentropy(mu, var, alpha, gamma=1):

    # var = logvar.exp()
    # var is sigma squared
    # alpha_var = log_alpha_var.exp()
    # print(f'alpha_var is {alpha_var}')
    # var = (alpha_var - 1) / (alpha - 1)
    # print(f'var is {var}')

    inside_log = (gamma + var * (alpha -1)) / gamma

    # print(f'var is {torch.sum(var)}')

    #
    # print(f'var sum is {torch.sum(var)}')
    # print(f'inside_log sum {torch.sum(inside_log)}')

    term1 = -torch.log(inside_log)
    # term1 = -log_alpha_var
    # print(f'term1 sum is {torch.sum(term1)}')

    # term2 = (1-alpha) * gamma ** 64 * torch.log(2*3.14159)**64
    term3 = -mu ** 2 / var
    term3 = torch.nan_to_num(term3)
    # print(f'term3 sum is {torch.sum(term3)}')

    term4 = mu **2 / var **2 * var * gamma / (gamma + var * (alpha-1))
    term4 = torch.nan_to_num(term4)
    # print(f'term4 sum is {torch.sum(term4)}')


    total = term1 + term3 + term4

    # print(f'return sum is {torch.sum(total)/(2-2*alpha) }')


    return torch.sum(total) / (2-2*alpha)

# IB or CFB loss
def get_KLdivergence_loss(yhat, y, mu, var, beta):

    divergence = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)
    # divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    yhat = torch.where(torch.isnan(yhat), torch.zeros_like(yhat), yhat)
    yhat = torch.where(torch.isinf(yhat), torch.zeros_like(yhat), yhat)

    cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(yhat.view(-1), y,
                                                                         reduction='sum')
    return divergence + beta * cross_entropy


# RFIB loss
def get_RFIB_loss(yhat, yhat_fair, y, mu, logvar, alpha, beta1, beta2):
    if alpha == 0:
        divergence = 0
    elif alpha == 1:  # KL Divergence
        divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    elif alpha < 1: # Renyi cross entropy
        divergence = renyi_crossentropy(mu, logvar, alpha)
    else: # Renyi divergence
        divergence = renyi_divergence(mu, logvar, alpha)

    IB_cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(yhat.view(-1), y,
                                                                            reduction='sum')
    CFB_cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(yhat_fair.view(-1), y,
                                                                             reduction='sum')

    # print(f'IB_cross_entropy is {IB_cross_entropy}')
    # print(f'beta1 * IB_cross_entropy is {beta1 * IB_cross_entropy}')
    # print(f'CFB_cross_entropy is {CFB_cross_entropy}')
    # print(f'beta2 * CFB_cross_entropy is {beta2 * CFB_cross_entropy}')

    loss = divergence + beta1 * IB_cross_entropy + beta2 * CFB_cross_entropy

    return loss
