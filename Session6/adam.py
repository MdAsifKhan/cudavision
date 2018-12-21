from copy import copy
import torch

class Adam:
    def __init__(self, alpha, beta1, beta2, eps, use_gpu=False):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.use_gpu = use_gpu

    def adam(self, gradient_t, theta_t_prev, convergence):
        t = 0
        m_t_prev, v_t_prev = torch.zeros_like(gradient_t), torch.zeros_like(gradient_t)
        if self.use_gpu:
            m_t_prev, v_t_prev, t = m_t_prev.cuda(), v_t_prev.cuda(), t.cuda() 

        while True:
            t += 1 
            m_t = beta1*m_t_prev + (1-beta1)*gradient_t
            v_t = beta2*v_t_prev + (1-beta2)*gradient_t**2
            m_hat = m_t/(1-beta1**t)
            v_hat = v_t/(1-beta2**t)
            theta_t = theta_t_prev - alpha*(m_hat/v_hat.sqrt().add_(eps))
            convergence.append(torch.norm(theta_t-theta_t_prev))
            if torch.norm(theta_t-theta_t_prev)<1e-4:
                break
            theta_t_prev = copy(theta_t)
        return theta_t_prev, convergence

    def run(self, gradient_t, theta_t_prev, convergence):
        theta_t, convergence = self.adam(gradient_t, theta_t_prev, convergence)
        return theta_t, convergence

if __name__ == '__main__':
	# Some x
	x = torch.randn(10, 4, requires_grad=False)
	# Some polynomial
	y_true = 2*x**2 - 5*x + 3
	#Initial Guess
	w1 = torch.randn(4, 1, requires_grad=True)
	use_gpu = False
	if use_gpu:
		w1, y, x = w1.cuda(), y.cuda(), x.cuda()

	epochs = 10
	alpha = 0.0001
	beta1 = 0.9
	beta2 = 0.9991
	eps = 1e-8
	optimizer = Adam(alpha, beta1, beta2, eps)
	for epoch in range(epochs):
		y_p = x.mm(w1)
		loss = (y_p - y_true).pow(2).sum()
		loss.backward(retain_graph=True)
		gradient_t = w1.grad
		w2 = optimizer.run(gradient_t, w1)