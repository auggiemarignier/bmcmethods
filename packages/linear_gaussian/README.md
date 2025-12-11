# Analytical Gaussian With Linear Mappings

In the linear case with Gaussian likelihoods and priors, we can write down the evidence analytically.

In general, for a product of Gaussians of a multivariate random variable $x \in \mathbb{R}^{n}$ we have

$$f(x) = \prod_{i=1}^N \mathcal{N}(A_i x | \mu_i, C_i) $$

for linear transformations $A_i \in \mathbb{R}^{n_i \times n}$, mean vectors $\mu_i \in \mathbb{R}^{n_i}$ and covariance matrices $C_i \in \mathbb{R}^{n_i \times n_i}$.

It can be shown that

$$ Z = \int f(x)\,\mathrm{d}x
     = (2\pi)^{\frac{n - \sum n_i}{2}}
       \prod |C_i| ^{-\frac{1}{2}}
       |S|^{-\frac{1}{2}}
       \exp{\left\{
                -\frac{1}{2}(c - y^T S y)
            \right\}}$$

where

$$ c = \sum \mu_i^T C_i^{-1} \mu_i \in \mathbb{R} $$

$$ s^T = \sum \mu_i^T C_i^{-1} A_i \in \mathbb{R}^n $$

$$ S = \sum A_i^T C_i^{-1} A_i \in \mathbb{R}^{n \times n} $$

$$ y = S^{-T}s \in \mathbb{R}^n $$

and all sums and products are taken over $i = \{1 \dots N\}$.

In the typical case where our prior is the distance from a reference model

- $N=2$ (likelihood $i=1$ and prior $i=2$)
- $n$ is the length of the model vector
- $n_1$ is the length of the data vector
- $n_2 = n$
- $A_1 = G$ is the linear forward operator
- $A_2 = I$
- $\mu_1 = d$ is the data vector
- $\mu_2 = m_0$ is the reference model
- $C_1 = C_d$ is the data covariance
- $C_2 = C_m$ is the prior covariance
