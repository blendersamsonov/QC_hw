# 1. Computing relative phase of a single Q-bit

$ \Psi = 
\begin{pmatrix}
\sin\theta \\
\cos\theta e^{i\varphi}
\end{pmatrix}$ 

Let's calculate $\mathbf{H}\Psi$:


$\mathbf{H}\Psi = \frac{1}{\sqrt{2}}
\begin{pmatrix}
1 & 1 \\
1 & -1
\end{pmatrix}
\begin{pmatrix}
\sin\theta \\
\cos\theta e^{i\varphi}
\end{pmatrix}
= \frac{1}{\sqrt{2}}
\begin{pmatrix}
\sin\theta + \cos\theta e^{i\varphi} \\
\sin\theta - \cos\theta e^{i\varphi}
\end{pmatrix}
$

Let's also calculate probabilities $P_H(0)$ and $P_H(1)$ of the Q-bit being in the state $\begin{pmatrix}1\\0\end{pmatrix}$ or $\begin{pmatrix}0\\1\end{pmatrix}$ correspondingly

$ P_H(0, 1) = \frac{1}{2} {\left| \sin\theta \pm \cos\theta e^{i\varphi} \right|}^2 =
\frac{1 \pm 2\sin\theta\cos\theta\cos\varphi}{2}$

So

$\cos\varphi = \frac{P_H(0)-P_H(1)}{2\sin\theta\cos\theta}$

And we know that in the circuit without the $\mathbf{H}$-gate the probabilities $P(0)$ and $P(1)$ are:

$P(0) = \sin^2\theta \\
P(1) = \cos^2\theta$

So finally

$\cos\varphi=\frac{P_H(0)-P_H(1)}{2\sqrt{P(0)P(1)}}$


```python
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram, plot_bloch_vector
import numpy as np
import numpy.random as rand

backend = Aer.get_backend('statevector_simulator')
```


```python
import pandas as pd
real_phase = []
eval_phase = []
for j in range(10):
    # Random initial vector
    theta, phi = rand.rand(2) * np.pi
    sin = np.sin(theta)
    cos = np.cos(theta)
    phase = np.exp(1j*phi)

    res = []
    for i in range(2):
        qc = QuantumCircuit(1)
        initial_state = [cos,sin*phase]
        qc.initialize(initial_state, 0)
        if i == 0:
            qc.h(0)
        qc.draw()
        res.append(execute(qc,backend).result().get_counts()) # return probabilities P(0) and P(1)
    phi_exp = (res[0]['0'] - res[0]['1']) / (2 * np.sqrt(res[1]['0'] * res[1]['1']))
    real_phase.append(np.cos(phi))
    eval_phase.append(phi_exp)
df = pd.DataFrame(data = {'Real':real_phase,'Evaluated':eval_phase})
df.index.name = 'Exp. №'
df['Relat. error'] = np.abs((np.abs(df['Real']) - np.abs(df['Evaluated'])) / np.abs(df['Real']))
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Real</th>
      <th>Evaluated</th>
      <th>Relat. error</th>
    </tr>
    <tr>
      <th>Exp. №</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.612683</td>
      <td>-0.612683</td>
      <td>1.449654e-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.374461</td>
      <td>-0.374461</td>
      <td>2.371884e-15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.203455</td>
      <td>0.203455</td>
      <td>1.227791e-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.353888</td>
      <td>-0.353888</td>
      <td>1.098026e-15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.491891</td>
      <td>-0.491891</td>
      <td>3.724132e-15</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.928190</td>
      <td>-0.928190</td>
      <td>5.980583e-16</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.963957</td>
      <td>0.963957</td>
      <td>5.413153e-15</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.464893</td>
      <td>0.464893</td>
      <td>1.194063e-15</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.617087</td>
      <td>-0.617087</td>
      <td>6.117063e-15</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.150843</td>
      <td>0.150843</td>
      <td>3.496061e-15</td>
    </tr>
  </tbody>
</table>
</div>



We see that the absolute values of the real and the evaluated phase are indeed equal; the sign though can not be defined in our experiment

# 2. Single Qubit state preparation

In terms of Bloch's sphere to prepare a general Qubit state from the $\begin{pmatrix}(1,0)\end{pmatrix}$ on could first rotate the vector around the $X$ axis on the angle $\pi/2$,
then rotate along the $Z$ axis on the angle $\theta$, rotate it back, i.e. around the $X$ axis on the angle $-\pi/2$ and then finally add a phase shift via rotation around the $Z$ axis on the angle $\varphi$
So let's check how the combination $\mathbf {R_z(\varphi)HR_z(2\theta)H}$ acts on the vector $\begin{pmatrix}1 \\ 0\end{pmatrix}$

$ \begin{pmatrix} 1 & 1 \\ 1 & e^{i\phi} \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 1 & e^{2i\theta} \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \begin{pmatrix} 1 \\ 0\end{pmatrix} = \begin{pmatrix} \frac{1-e^{2i\theta}}{2}  \\ e^{i\phi}\frac{1+e^{2i\theta}}{2} \end{pmatrix} = 
e^{i\theta}\begin{pmatrix}i\sin\theta \\ \cos\theta e^{i\phi} \end{pmatrix}$

Let's also multiply by $\mathbf{S}$ to illiminate imaginary unit in $a_0$, so to get the desired state we use the operator $\mathbf {U_3 = S R_z(\varphi)HR_z(2\theta)H}$

Let's compare to Qiskit's built-in $\mathbf{U_3}$


```python
# Random initial vector
theta, phi = rand.rand(2) * np.pi

res = []
for i in range(2):
    qc = QuantumCircuit(1)
    qc.initialize([1,0], 0)
    if i == 0:
        qc.u3(2*theta, phi, 0, 0) # Qiskit's U_3
        res = execute(qc,backend).result().get_statevector()
        q_re0 = np.real(res[0])
        q_im0 = np.imag(res[0])
        q_re1 = np.real(res[1])
        q_im1 = np.imag(res[1])
    else: 
        qc.h(0)
        qc.rz(2*theta,0)
        qc.h(0)          # My implementation
        qc.rz(phi, 0)
        qc.s(0)
        res = execute(qc,backend).result().get_statevector() * np.power(np.e, - 1j * theta) # return statevector multiplied by the phase
        my_re0 = np.real(res[0])
        my_im0 = np.imag(res[0])
        my_re1 = np.real(res[1])
        my_im1 = np.imag(res[1])
```


```python
print(f" a_0 real || qiskit : {q_re0:.2e} || my : {my_re0:.2e} || delta : {2*np.abs(q_re0 - my_re0):.2e}")
print(f" a_0 imag || qiskit : {q_im0:.2e} || my : {my_im0:.2e} || delta : {2*np.abs(q_im0 - my_im0):.2e}")
print(f" a_1 real || qiskit : {q_re1:.2e} || my : {my_re1:.2e} || delta : {2*np.abs(q_re1 - my_re1):.2e}")
print(f" a_1 imag || qiskit : {q_im1:.2e} || my : {my_im1:.2e} || delta : {2*np.abs(q_im1 - my_im1):.2e}")
```

     a_0 real || qiskit : 4.73e-01 || my : 4.73e-01 || delta : 2.22e-16
     a_0 imag || qiskit : 0.00e+00 || my : -1.94e-16 || delta : 3.89e-16
     a_1 real || qiskit : 8.60e-01 || my : 8.60e-01 || delta : 2.22e-16
     a_1 imag || qiskit : 1.92e-01 || my : 1.92e-01 || delta : 1.11e-16


Relative errors are negligibly small

# 3. Useful identities

Proof by direct comparison

### 1) 
$\mathbf{HXH} = 
\frac{1}{2}
\begin{pmatrix} 
1 & 1 \\
1 & -1
\end{pmatrix}
\begin{pmatrix} 
0 & 1 \\
1 & 0
\end{pmatrix}
\begin{pmatrix} 
1 & 1 \\
1 & -1
\end{pmatrix}=
\frac{1}{2}
\begin{pmatrix} 
1 & 1 \\
-1 & 1
\end{pmatrix}
\begin{pmatrix} 
1 & 1 \\
1 & -1
\end{pmatrix}=
\begin{pmatrix} 
1 & 0 \\
0 & -1
\end{pmatrix}
\equiv\mathbf{Z}$

### 2) 
$\mathbf{cZ_0} = 
\begin{pmatrix}
I & 0 \\
0 & Z
\end{pmatrix} =
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & -1
\end{pmatrix}$


$\mathbf{cZ_1} = 
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & \mathbf{Z}_{00} & 0 & \mathbf{Z}_{01} \\
0 & 0 & 1 & 0 \\
0 & \mathbf{Z}_{10} & 0 & \mathbf{Z}_{11}
\end{pmatrix}=\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & -1
\end{pmatrix}=\mathbf{cZ_0}$

### 3) 
$\left(\mathbf{H\otimes H}\right) \mathbf{cX} \left(\mathbf{H\otimes H}\right)=?$

a) $\mathbf{H\otimes H} = \frac{1}{2}
\begin{pmatrix}
1 & 1 & 1 & 1 \\
1 & -1 & 1 & -1 \\
1 & 1 & -1 & -1 \\
1 & -1 & -1 & 1 \\
\end{pmatrix}$


b) $\frac{1}{4}
\begin{pmatrix}
1 & 1 & 1 & 1 \\
1 & -1 & 1 & -1 \\
1 & 1 & -1 & -1 \\
1 & -1 & -1 & 1 \\
\end{pmatrix}
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0 \\
\end{pmatrix}
\begin{pmatrix}
1 & 1 & 1 & 1 \\
1 & -1 & 1 & -1 \\
1 & 1 & -1 & -1 \\
1 & -1 & -1 & 1 \\
\end{pmatrix}=\frac{1}{4}
\begin{pmatrix}
1 & 1 & 1 & 1 \\
1 & -1 & -1 & 1 \\
1 & 1 & -1 & -1 \\
1 & -1 & 1 & -1 \\
\end{pmatrix}
\begin{pmatrix}
1 & 1 & 1 & 1 \\
1 & -1 & 1 & -1 \\
1 & 1 & -1 & -1 \\
1 & -1 & -1 & 1 \\
\end{pmatrix}=
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 1 \\
0 & 0 & 1 & 0 \\
0 & 1 & 0 & 0 \\
\end{pmatrix}\equiv
\mathbf{cX_1}=
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & \mathbf{X}_{00} & 0 & \mathbf{X}_{01} \\
0 & 0 & 1 & 0 \\
0 & \mathbf{X}_{10} & 0 & \mathbf{X}_{11}
\end{pmatrix}$

### 4) 
$\mathbf{c{e^{i\alpha}}_0} = 
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & e^{i\alpha} & 0 \\
0 & 0 & 0 & e^{i\alpha}
\end{pmatrix}$
$U_1(\alpha)\otimes I=
\begin{pmatrix}
1 & 0 \\
0 & e^{i\alpha}
\end{pmatrix}\otimes
\begin{pmatrix}
1 & 0 \\
0 & 1
\end{pmatrix}=
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & e^{i\alpha} & 0 \\
0 & 0 & 0 & e^{i\alpha}
\end{pmatrix}$

# 4. Separable and entangled states

## (a). $ |\psi\rangle = \frac{2}{3} |00\rangle + \frac{1}{3} |01\rangle - \frac{2}{3}|11\rangle $
If the state is separable, i.e. $|\psi\rangle=|a\rangle \otimes |b\rangle$ where $|a\rangle = \begin{pmatrix}a_0 \\ a_1\end{pmatrix}$, $|b\rangle = \begin{pmatrix}b_0 \\ b_1\end{pmatrix}$, then the following system of equations has a solution:

$\begin{align}
a_0 b_0 =& 2/3  &(1) \\
a_0 b_1 =& 1/3  &(2) \\
a_1 b_0 =& 0    &(3) \\
a_1 b_1 =& -2/3 &(4) \\
\end{align}$

Eqs. (1), (3) and (4) cannot be satisfied together, so the system does not have a solution and the state is **entangled**

## (b). $ |\psi\rangle = \frac{1}{2}\left(|00\rangle - i |01\rangle + i |10\rangle + |11\rangle \right)$

$\begin{align}
a_0 b_0 =& 1/2  &(1) \\
a_0 b_1 =& -i/2  &(2) \\
a_1 b_0 =& i/2    &(3) \\
a_1 b_1 =& 1/2 &(4) \\
\end{align}$

The system have a solution $a_0 = -i/\sqrt{2}$, $a_1 = 1/\sqrt{2}$, $b_0 = i/\sqrt{2}$, $b_1 = 1/\sqrt{2}$ so the state is **separable**

## (c). $ |\psi\rangle = \frac{1}{2}\left(|00\rangle - |01\rangle + |10\rangle + |11\rangle \right)$

$\begin{align}
a_0 b_0 =& 1/2  &(1) \\
a_0 b_1 =& -1/2  &(2) \\
a_1 b_0 =& 1/2    &(3) \\
a_1 b_1 =& 1/2 &(4) \\
\end{align}$

It follows from (3) and (4) that $b_0=b_1$, but it leads to the fact that Eqs. (1) and (2) are not compatible. So the system does not have a solution and the state is **entangled**

# 6. Measuring gate

## $|\psi\rangle = \sum\limits_x \alpha_x |x\rangle_m |\Phi_x\rangle_{n-m}$

We can rewrite $\Phi_x$ in the same manner:

## $|\Phi_x\rangle = \sum\limits_y \beta_y |y\rangle_k |\Theta_{x,y}\rangle_{n-m-k}$

So after the measurment of the first $m$ qbits we get the state $|\Phi_x\rangle$ with the probability $\alpha_x$. Then we measure the next $k$ qbits and get state $|\Theta_{x,y}\rangle$ with the conditional probability $\beta_y$. So the total probability of the state $|\Theta_{x,y}\rangle$ is $\alpha_x \cdot \beta_y$

But if we subsitute $\Phi_x$ with the new expression:

## $|\psi\rangle = \sum\limits_x \alpha_x |x\rangle_m \sum\limits_y \beta_y |y\rangle_k |\Theta_{x,y}\rangle_{n-m-k}$

it is obvious that the probability of measuring the system in the state $|\Theta_{x,y}\rangle$ is equal to $\alpha_x \cdot \beta_y$, Q.E.D.
