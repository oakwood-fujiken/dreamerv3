# DreamerV3 セットアップ & モデル学習ガイド

DreamerV3 は世界モデルベースの強化学習アルゴリズムで、固定のハイパーパラメータで多様なドメインを学習できる。
本ドキュメントでは環境構築からモデルの学習・評価までの手順をまとめる。

---

## 1. 環境構築

### 1.1 Rust のインストール

DreamerV3 の一部コンポーネント (`dreamerv3-rust/`) は Rust で実装されている。
まず Rust ツールチェインをインストールする。

**Linux / macOS:**

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

**インストール確認:**

```bash
rustc --version
cargo --version
```

### 1.2 Python 依存パッケージのインストール

Python 3.11 以上が必要。

```bash
pip install -U -r requirements.txt
```

主な依存パッケージ:

| パッケージ | 用途 |
|---|---|
| `jax[cuda12]` | GPU 上でのモデル計算 |
| `optax` | オプティマイザ |
| `elements` | コアユーティリティ |
| `ninjax` | JAX ベースの NN フレームワーク |
| `ale_py` | Atari 環境 |
| `scope` | メトリクス可視化 |

### 1.3 Docker を使う場合

全環境（Atari, DMC, Crafter, Minecraft 等）がプリインストールされた Docker イメージを利用できる。

```bash
docker build -f Dockerfile -t dreamerv3 .
docker run -it --rm --gpus all \
  -v ~/logdir:/logdir \
  dreamerv3 \
  python dreamerv3/main.py \
    --logdir /logdir/{timestamp} \
    --configs crafter \
    --task crafter_reward
```

### 1.4 環境ごとの追加インストール

使用するタスクに応じて追加パッケージが必要になる場合がある。

```bash
# Crafter
pip install crafter

# DeepMind Control Suite
pip install dm_control

# Procgen
pip install procgen

# Minecraft (MineRL)
pip install minerl
```

詳細は `Dockerfile` 内のインストール手順を参照。

---

## 2. プロジェクト構成

```
dreamerv3/
├── dreamerv3/
│   ├── main.py          # エントリポイント
│   ├── agent.py         # エージェント（エンコーダ, RSSM, デコーダ, Actor-Critic）
│   ├── rssm.py          # Recurrent State Space Model（世界モデル）
│   └── configs.yaml     # 設定ファイル（全ハイパーパラメータ）
├── embodied/
│   ├── run/             # 学習ループ（train, train_eval, eval_only, parallel）
│   ├── core/            # Driver, Replay Buffer, Wrappers
│   ├── envs/            # 環境ラッパー（atari, dmc, crafter 等）
│   └── jax/             # JAX ベースの NN 実装
├── dreamerv3-rust/      # Rust ブリッジ実装
├── requirements.txt
├── Dockerfile
└── baselines.yaml       # ベンチマークスコア
```

---

## 3. モデル学習の実行

### 3.1 基本コマンド

```bash
python dreamerv3/main.py \
  --logdir ~/logdir/{timestamp} \
  --configs <config名> \
  --task <タスク名>
```

- `--logdir`: ログ・チェックポイントの保存先。`{timestamp}` は自動展開される
- `--configs`: `configs.yaml` 内の設定ブロック名（複数指定可）
- `--task`: 実行するタスク名

### 3.2 動作確認（デバッグ実行）

まずは小さなモデルで動作確認する:

```bash
python dreamerv3/main.py \
  --logdir ~/logdir/debug \
  --configs debug \
  --task dummy_disc
```

`debug` コンフィグはネットワークサイズとログ間隔を縮小し、高速に動作確認できる。

### 3.3 タスク別の学習コマンド例

**Crafter:**

```bash
python dreamerv3/main.py \
  --logdir ~/logdir/crafter/{timestamp} \
  --configs crafter \
  --task crafter_reward \
  --run.train_ratio 512
```

**Atari (例: Pong):**

```bash
python dreamerv3/main.py \
  --logdir ~/logdir/atari/{timestamp} \
  --configs atari \
  --task atari_pong
```

**DeepMind Control Suite (画像観測):**

```bash
python dreamerv3/main.py \
  --logdir ~/logdir/dmc/{timestamp} \
  --configs dmc_vision \
  --task dmc_walker_walk
```

**DeepMind Control Suite (状態観測):**

```bash
python dreamerv3/main.py \
  --logdir ~/logdir/dmc/{timestamp} \
  --configs dmc_proprio \
  --task dmc_cheetah_run
```

**Minecraft:**

```bash
python dreamerv3/main.py \
  --logdir ~/logdir/minecraft/{timestamp} \
  --configs minecraft \
  --task minecraft_diamond
```

---

## 4. 設定システム

### 4.1 設定の仕組み

全設定は `dreamerv3/configs.yaml` に定義されている。
設定は以下の優先順位で上書きされる:

```
defaults（基本設定） → configs 引数で指定したブロック → CLI フラグ
```

複数の設定ブロックを組み合わせることも可能:

```bash
--configs crafter size50m debug
```

### 4.2 主要な設定項目

| 設定 | デフォルト | 説明 |
|---|---|---|
| `batch_size` | 16 | バッチサイズ |
| `batch_length` | 64 | シーケンス長 |
| `run.steps` | 1e10 | 総学習ステップ数 |
| `run.train_ratio` | 32.0 | 環境ステップあたりの学習回数 |
| `run.envs` | 16 | 並列環境数 |
| `agent.opt.lr` | 4e-5 | 学習率 |
| `jax.platform` | cuda | デバイス（`cuda` / `cpu` / `tpu`） |
| `jax.compute_dtype` | bfloat16 | 計算精度 |

### 4.3 モデルサイズの変更

事前定義されたサイズバリアントを指定できる:

| コンフィグ名 | パラメータ数 |
|---|---|
| `size1m` | 1M（最小・テスト用） |
| `size12m` | 12M |
| `size25m` | 25M |
| `size50m` | 50M（デフォルト） |
| `size100m` | 100M |
| `size200m` | 200M |
| `size400m` | 400M |

```bash
# 200M パラメータモデルで学習
python dreamerv3/main.py \
  --logdir ~/logdir/large/{timestamp} \
  --configs atari size200m \
  --task atari_breakout
```

### 4.4 CLI からの設定上書き

`configs.yaml` 内の任意の設定を CLI フラグで上書きできる:

```bash
python dreamerv3/main.py \
  --logdir ~/logdir/custom/{timestamp} \
  --configs crafter \
  --task crafter_reward \
  --batch_size 8 \
  --batch_length 32 \
  --agent.opt.lr 1e-4 \
  --jax.platform cpu
```

---

## 5. 対応タスク一覧

| スイート | タスク例 | コンフィグ |
|---|---|---|
| Atari | `atari_pong`, `atari_breakout` 等 57 ゲーム | `atari` |
| Atari 100K | `atari100k_pong` | `atari100k` |
| Crafter | `crafter_reward`, `crafter_noreward` | `crafter` |
| DMC (画像) | `dmc_walker_walk`, `dmc_cheetah_run` | `dmc_vision` |
| DMC (状態) | `dmc_walker_walk`, `dmc_cheetah_run` | `dmc_proprio` |
| Procgen | `procgen_coinrun`, `procgen_maze` 等 16 ゲーム | `procgen` |
| DMLab | `dmlab_explore_goal_locations_small` 等 30 タスク | `dmlab` |
| Minecraft | `minecraft_diamond`, `minecraft_log` | `minecraft` |
| LocoNav | `loconav_ant_maze_m` | `loconav` |
| BSuite | `bsuite_mnist/0` | `bsuite` |
| Memory Maze | `memmaze_*` | `memmaze` |
| Dummy | `dummy_disc`, `dummy_cont` | (デフォルト) |

---

## 6. 学習の監視

### 6.1 Scope（推奨）

```bash
pip install -U scope
python -m scope.viewer --basedir ~/logdir --port 8000
# ブラウザで http://localhost:8000 を開く
```

### 6.2 ログ出力

学習中のメトリクスは以下に出力される:

- **ターミナル**: リアルタイムで主要メトリクスを表示
- **JSONL**: `<logdir>/metrics.jsonl`, `<logdir>/scores.jsonl`
- **Scope**: `<logdir>/` 内のサマリファイル

### 6.3 TensorBoard / WandB（オプション）

`main.py` 内のロガー設定を変更して有効化できる:

```bash
--logger.outputs [jsonl,scope,tensorboard]
--logger.outputs [jsonl,scope,wandb]
```

### 6.4 主要メトリクス

| メトリクス | 意味 |
|---|---|
| `episode/score` | エピソードの累積報酬 |
| `episode/length` | エピソード長 |
| `train/loss/dyn` | 世界モデル動態損失 |
| `train/loss/rec` | 再構成損失 |
| `train/loss/rew` | 報酬予測損失 |
| `fps/policy` | ポリシー実行の FPS |
| `fps/train` | 学習の FPS |

---

## 7. チェックポイントと学習の再開

### 7.1 自動保存

チェックポイントは `<logdir>/ckpt/` に定期的に自動保存される（デフォルト: 900秒ごと）。
エージェントの重み、オプティマイザ状態、リプレイバッファ、ステップカウンタが含まれる。

### 7.2 学習の再開

同じ `--logdir` で再実行すると、最新のチェックポイントから自動的に再開される:

```bash
# 前回と全く同じコマンドを実行するだけ
python dreamerv3/main.py \
  --logdir ~/logdir/crafter/20250101T120000 \
  --configs crafter \
  --task crafter_reward
```

### 7.3 チェックポイントからの読み込み

別のチェックポイントから重みを読み込む場合:

```bash
--run.from_checkpoint <チェックポイントのパス>
--run.from_checkpoint_regex <重みのフィルタ正規表現>
```

---

## 8. 学習モード

`main.py` は複数の学習モードをサポートしている:

| モード | 説明 |
|---|---|
| `train`（デフォルト） | 環境からデータ収集しながら学習 |
| `train_eval` | 学習と評価を分離して実行 |
| `eval_only` | チェックポイントを読み込んで評価のみ |
| `parallel` | 分散学習（複数プロセス） |

---

## 9. トラブルシューティング

| 問題 | 対処法 |
|---|---|
| OOM（メモリ不足） | `--batch_size 1` で縮小してテスト |
| PyTreeDef エラー | チェックポイントと設定が不整合。`logdir` を新規にする |
| CUDA エラー | ログを上にスクロールして根本原因を確認 |
| CPU で実行したい | `--jax.platform cpu` を指定 |
| 環境が見つからない | 対応パッケージをインストール（セクション 1.4 参照） |
| 学習が遅い | `--run.train_ratio` を下げる / `--run.envs` を調整 |

---

## 10. クイックスタートまとめ

```bash
# 1. 依存パッケージのインストール
pip install -U -r requirements.txt

# 2. 動作確認（CPU, ダミー環境）
python dreamerv3/main.py \
  --logdir ~/logdir/test \
  --configs debug \
  --task dummy_disc

# 3. Crafter で本格的な学習
pip install crafter
python dreamerv3/main.py \
  --logdir ~/logdir/crafter/{timestamp} \
  --configs crafter \
  --task crafter_reward \
  --run.train_ratio 512

# 4. 学習の監視
pip install -U scope
python -m scope.viewer --basedir ~/logdir --port 8000
```
