import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

# ==========================================
# 1. 各活性化関数の定義（数式の実装）
# ==========================================

def step_function(x):
    """1. ステップ関数
    0を境目にして、0未満なら「0」、0以上なら「1」をパキッと切り替えて出力します。
    """
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    """2. シグモイド関数
    どんな入力値でも、出力を必ず 0.0 〜 1.0 の間の滑らかな数値（確率）に変換します。
    """
    return 1 / (1 + np.exp(-x))

def relu(x):
    """3. ReLU（レルー）関数
    入力が0未満なら「完全に0」、0以上なら入力された値を「そのまま」出力します。
    現代の中間層（隠れ層）の主役です。
    """
    return np.maximum(0, x)

def softmax_for_graph(x):
    """4. ソフトマックス関数（グラフ描画用）
    本来は「複数の値の合計を1（100%）」にする関数です。
    ここではグラフで形を見るために「相手のスコアが0」と固定された状況で、
    自分のスコア x が変化したときの、自分の確率（0.0 〜 1.0）の変化を計算しています。
    """
    # exp(x) / (exp(x) + exp(0)) を計算しています（※exp(0)は1です）
    return np.exp(x) / (np.exp(x) + np.exp(0))


# ==========================================
# 2. グラフ描画用のデータ準備
# ==========================================

# -5 から 5 までの間を 100等分した、グラフの横軸（x軸）のデータを作成
x = np.linspace(-5, 5, 100)

# それぞれの関数に入力データ x を通して、縦軸（y軸）のデータを計算
y_step = step_function(x)
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_softmax = softmax_for_graph(x)


# ==========================================
# 3. Matplotlibによるグラフの描画（2×2の配置）
# ==========================================

# グラフ全体のサイズを設定（横10インチ、縦8インチ）
plt.figure(figsize=(10, 8))

# --- 左上：ステップ関数 ---
plt.subplot(2, 2, 1)
plt.plot(x, y_step, label="Step", color="blue", linewidth=2)
plt.title("1. ステップ関数 (Step Function)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6) # グリッド線
plt.axvline(0, color="black", linewidth=0.8) # 縦の0基準線
plt.axhline(0, color="black", linewidth=0.8) # 横の0基準線
plt.ylim(-0.2, 1.2) # y軸の表示範囲

# --- 右上：シグモイド関数 ---
plt.subplot(2, 2, 2)
plt.plot(x, y_sigmoid, label="Sigmoid", color="orange", linewidth=2)
plt.title("2. シグモイド関数 (Sigmoid Function)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.axvline(0, color="black", linewidth=0.8)
plt.axhline(0, color="black", linewidth=0.8)
plt.ylim(-0.2, 1.2)

# --- 左下：ReLU関数 ---
plt.subplot(2, 2, 3)
plt.plot(x, y_relu, label="ReLU", color="green", linewidth=2)
plt.title("3. ReLU関数 (Rectified Linear Unit)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.axvline(0, color="black", linewidth=0.8)
plt.axhline(0, color="black", linewidth=0.8)
plt.ylim(-1.0, 5.5) # ReLUは1以上にも伸びるため範囲を広げています

# --- 右下：ソフトマックス関数 ---
plt.subplot(2, 2, 4)
plt.plot(x, y_softmax, label="Softmax", color="red", linewidth=2)
plt.title("4. ソフトマックス関数 (Softmax ※2クラス時)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.axvline(0, color="black", linewidth=0.8)
plt.axhline(0, color="black", linewidth=0.8)
plt.ylim(-0.2, 1.2)

# レイアウトを綺麗に自動調整して画面に表示
plt.tight_layout()
plt.savefig("13-04.png")