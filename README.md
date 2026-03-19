# DHM theta/phi 스윕 (웹 버전)

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
streamlit run app.py
```

실행 후 브라우저에서 Streamlit 페이지가 열립니다.

## 기능

- **Theta 스윕**: theta 범위/스텝 지정, phi는 고정
- **Phi 스윥**: phi 범위/스텝 지정, theta는 고정
- **Theta×Phi 그리드**: theta와 phi를 동시에 스윕(조합)

> 참고: 이 데모는 단순 backprojection 기반 “초기 추정”이라 각도 수가 적으면 아티팩트가 큽니다. 더 원본에 가까운 복원을 원하면 각도 개수를 늘리거나 반복 복원(ART/SART/정규화)을 추가해야 합니다.

## 수학적 정의(현재 앱 구현)

### 좌표/각도 정의 (표준 구면좌표계)

- **부피 데이터(볼륨)**: \(f(x,y,z)\) 를 정육면체 영역 \([-1,1]^3\) 위에 이산 샘플링한 3D 배열로 둡니다.
- **각도**
  - \(\theta\): +z축에서 내려오는 polar 각 (\(0^\circ \le \theta \le 180^\circ\))
  - \(\phi\): x–y 평면의 azimuth 각 (+x에서 +y 방향, \(0^\circ \le \phi < 360^\circ\))  
    (구현상 내부에서는 \(\phi\)에 대해 \(360^\circ\) 주기(mod \(360^\circ\))만 사용하므로, \(-180^\circ\sim180^\circ\) 형태의 입력도 모두 같은 방향으로 해석됩니다.)

이에 대응하는 **관측(광선) 방향 단위벡터**를

\[
\mathbf{v}(\theta,\phi)=
\begin{bmatrix}
\sin\theta\cos\phi\\
\sin\theta\sin\phi\\
\cos\theta
\end{bmatrix}
\]

로 둡니다(각도는 radian으로 환산해 사용).

### 투영(phase map) 정의

앱에는 2가지 투영 방식이 있습니다.

#### 1) 회전 후 z합(빠름)

\(\mathbf{v}(\theta,\phi)\) 를 +z축 \(\mathbf{e}_z\) 로 보내는 회전행렬 \(R(\theta,\phi)\) (즉 \(R\mathbf{v}=\mathbf{e}_z\))를 만들고,
회전된 볼륨을

\[
f_R(\mathbf{x}) = f(R^{-1}\mathbf{x})
\]

로 정의하면, 2D phase map은

\[
P_{\text{rot}}(x,y;\theta,\phi)=\int f_R(x,y,z)\,dz
\approx \sum_{k} f_R(x,y,z_k)
\]

처럼 **z축으로 적분(이산에서는 z축 합)** 한 것으로 구현되어 있습니다.

> 구현 메모: 실제 코드는 \(f_R\)를 `affine_transform`으로 보간해 만든 뒤 z축으로 `sum`합니다.

#### 2) 광선 선적분(정확, 느림)

물체는 고정하고, 각 픽셀(검출기 평면 좌표)에서 \(\mathbf{v}\) 방향으로 선적분합니다.
검출기 평면은 \(\mathbf{v}\)에 수직인 평면으로 두고, 그 위의 직교기저 \(\mathbf{u},\mathbf{w}\) (\(\mathbf{u}\perp\mathbf{v}\), \(\mathbf{w}\perp\mathbf{v}\), \(\mathbf{w}=\mathbf{v}\times\mathbf{u}\))를 잡으면,

\[
P_{\text{ray}}(a,b;\theta,\phi)=\int f\!\left(a\,\mathbf{u}+b\,\mathbf{w}+t\,\mathbf{v}\right)\,dt
\]

를 계산합니다. 앱에서는 \(t\in[-\sqrt{3},\sqrt{3}]\) 범위를 균일 샘플링해 **사다리꼴에 가까운 단순 합**으로 적분합니다.

### (초기) Backprojection 정의

이 앱의 3D 결과는 “정확 복원”이 아니라 **backprojection 기반 초기 추정**입니다.

- **회전 후 z합 방식의 backprojection**: 2D phase를 z로 복제한 뒤(“기둥”), \(R^{-1}\)로 역회전합니다.

\[
B_{\text{rot}}(\mathbf{x};\theta,\phi)\approx \left[\text{blur}_z\left(P_{\text{rot}}(\cdot,\cdot;\theta,\phi)\ \text{를 z로 복제}\right)\right]\circ R^{-1}
\]

- **광선 선적분 방식의 backprojection(데모용 근사)**: 각 픽셀 값을 해당 광선 위의 격자점들에 분배(현재 구현은 최근접 voxel 누적)합니다.

여러 각도 \(\{(\theta_i,\phi_i)\}\)에 대해 backprojection을 평균내어 초기 볼륨을 만듭니다.

\[
V_{\text{init}}(\mathbf{x}) = \frac{1}{M}\sum_{i=1}^{M} B(\mathbf{x};\theta_i,\phi_i)
\]

> 참고: 이는 각 방향 결과의 “교집합(AND)”이 아니라, 같은 좌표계에서의 **가중 합/평균(soft accumulation)** 입니다. 여러 방향에서 동시에 지지되는 위치는 값이 커져 공통영역처럼 보일 수 있지만, 수학적으로는 불리언 교집합이 아닙니다.

### 초기 추정 볼륨에서 z축(깊이) 크기 해석

초기 추정 \(V_{\text{init}}\)에서 “z축 방향으로 얼마나 두껍게 보이느냐”는, **물리적으로 유일하게 결정되는 값이라기보다** 앱이 채택한 *backprojection 규칙*과 *이산 격자 정의*에 의해 정해집니다.

- **회전 후 z합(빠름)**:
  - 2D phase map \(P(x,y)\)를 3D로 만들 때 \(z\) 방향으로 **그대로 복제**하여 \(V(x,y,z)=P(x,y)\) 형태의 “기둥(column)”을 만든 뒤 역회전합니다.
  - 따라서 z축의 길이는 **볼륨 격자 크기 \(N\)** (즉 \(N\times N\times N\)에서의 z축 샘플 수)로 정해집니다.
  - 시각적으로 퍼짐(두께감)은 `Backprojection z-blur sigma` 및 각도 분포에 크게 좌우됩니다.

- **광선 선적분(정확, 느림)**:
  - 2D phase의 값을 해당 광선 방향 \(\mathbf{v}\)를 따라 볼륨 내부로 분배합니다.
  - 앱은 볼륨 공간을 \([-1,1]^3\)로 두고, 광선이 큐브 전체를 관통하도록 \(t\in[-\sqrt{3},\sqrt{3}]\) 범위를 샘플링합니다(원점에서 큐브 꼭짓점까지 최대 거리 = \(\sqrt{3}\)).
  - 따라서 “깊이 방향 범위”는 **볼륨의 공간 범위(여기서는 \([-1,1]^3\))** 에 의해 결정됩니다.

> 주의: \(V_{\text{init}}\)은 교육용 “초기 추정”이며, 각도 수가 적으면 streak/늘어짐 아티팩트가 크게 나타납니다.

