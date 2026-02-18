# Rust インストール & 基本学習ガイド

## 1. インストール

### 1.1 rustup を使ったインストール（推奨）

`rustup` は Rust の公式インストーラ兼バージョン管理ツール。

**Linux / macOS:**

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

インストール後、シェルの環境変数を反映する:

```bash
source "$HOME/.cargo/env"
```

**Windows:**

[https://rustup.rs](https://rustup.rs) から `rustup-init.exe` をダウンロードして実行する。
事前に Visual Studio C++ Build Tools が必要。

### 1.2 インストールの確認

```bash
rustc --version    # コンパイラ
cargo --version    # パッケージマネージャ / ビルドツール
rustup --version   # ツールチェイン管理
```

### 1.3 アップデートとアンインストール

```bash
rustup update           # 最新版に更新
rustup self uninstall   # アンインストール
```

### 1.4 ツールチェインの管理

```bash
rustup toolchain list               # インストール済み一覧
rustup default stable               # stable を既定に設定
rustup toolchain install nightly     # nightly をインストール
```

---

## 2. 最初のプロジェクト

### 2.1 Cargo でプロジェクトを作成

```bash
cargo new hello_rust    # バイナリプロジェクト
cd hello_rust
```

生成されるファイル構成:

```
hello_rust/
├── Cargo.toml    # プロジェクト設定・依存関係
└── src/
    └── main.rs   # エントリポイント
```

### 2.2 ビルドと実行

```bash
cargo build          # デバッグビルド (target/debug/)
cargo build --release  # リリースビルド (target/release/)
cargo run            # ビルド + 実行
cargo check          # コンパイル可能か高速チェック（バイナリ生成なし）
```

---

## 3. Rust の基本文法

### 3.1 変数と可変性

```rust
let x = 5;           // 不変（デフォルト）
let mut y = 10;      // 可変
y = 20;              // OK

const MAX: u32 = 100; // コンパイル時定数
```

Rust では変数はデフォルトで不変。`mut` を付けて明示的に可変にする。

### 3.2 基本的な型

```rust
// 整数
let a: i32 = -42;     // 符号付き (i8, i16, i32, i64, i128)
let b: u64 = 100;     // 符号なし (u8, u16, u32, u64, u128)

// 浮動小数点
let c: f64 = 3.14;

// 真偽値
let d: bool = true;

// 文字・文字列
let e: char = 'あ';           // Unicode 1文字
let f: &str = "hello";        // 文字列スライス
let g: String = String::from("hello");  // ヒープ上の文字列

// タプルと配列
let tup: (i32, f64, char) = (1, 2.0, 'a');
let arr: [i32; 3] = [1, 2, 3];
```

### 3.3 関数

```rust
fn add(a: i32, b: i32) -> i32 {
    a + b   // 最後の式が戻り値（セミコロンなし）
}

fn greet(name: &str) {
    println!("Hello, {}!", name);
}
```

### 3.4 制御フロー

```rust
// if 式（値を返せる）
let number = 7;
let label = if number > 5 { "big" } else { "small" };

// loop
let mut count = 0;
loop {
    count += 1;
    if count == 5 { break; }
}

// while
while count > 0 {
    count -= 1;
}

// for
for i in 0..5 {
    println!("{}", i);   // 0, 1, 2, 3, 4
}

for item in vec![1, 2, 3].iter() {
    println!("{}", item);
}
```

### 3.5 所有権（Ownership）

Rust の最も重要な概念。メモリ安全性をコンパイル時に保証する仕組み。

**3つのルール:**

1. 各値には所有者（変数）が1つだけ存在する
2. 所有者がスコープを抜けると値は破棄される
3. 所有権は移動（move）される

```rust
let s1 = String::from("hello");
let s2 = s1;       // s1 の所有権が s2 に移動
// println!("{}", s1);  // コンパイルエラー: s1 はもう使えない

let s3 = s2.clone();   // 明示的なコピー（深いコピー）
println!("{} {}", s2, s3);  // どちらも使える
```

### 3.6 参照と借用（References & Borrowing）

所有権を移動せずに値を使う方法。

```rust
fn length(s: &String) -> usize {   // 不変参照（借用）
    s.len()
}

fn append(s: &mut String) {        // 可変参照
    s.push_str(" world");
}

let mut text = String::from("hello");
let len = length(&text);        // 不変参照を渡す
append(&mut text);               // 可変参照を渡す
```

**借用のルール:**

- 不変参照は同時にいくつでも持てる
- 可変参照は同時に1つだけ
- 不変参照と可変参照は同時に存在できない

### 3.7 構造体（Struct）

```rust
struct User {
    name: String,
    age: u32,
    active: bool,
}

impl User {
    // 関連関数（コンストラクタ相当）
    fn new(name: &str, age: u32) -> Self {
        User {
            name: String::from(name),
            age,
            active: true,
        }
    }

    // メソッド
    fn greet(&self) {
        println!("I'm {}, {} years old.", self.name, self.age);
    }
}

let user = User::new("Taro", 30);
user.greet();
```

### 3.8 列挙型（Enum）と パターンマッチ

```rust
enum Color {
    Red,
    Green,
    Blue,
    Custom(u8, u8, u8),
}

let c = Color::Custom(255, 128, 0);

match c {
    Color::Red => println!("red"),
    Color::Green => println!("green"),
    Color::Blue => println!("blue"),
    Color::Custom(r, g, b) => println!("rgb({}, {}, {})", r, g, b),
}
```

標準ライブラリの重要な列挙型:

```rust
// Option: 値があるかないか
let some_val: Option<i32> = Some(42);
let no_val: Option<i32> = None;

if let Some(v) = some_val {
    println!("value: {}", v);
}

// Result: 成功か失敗か
fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err(String::from("division by zero"))
    } else {
        Ok(a / b)
    }
}

match divide(10.0, 3.0) {
    Ok(result) => println!("{}", result),
    Err(e) => println!("Error: {}", e),
}
```

### 3.9 エラーハンドリング（? 演算子）

```rust
use std::fs;

fn read_file(path: &str) -> Result<String, std::io::Error> {
    let content = fs::read_to_string(path)?;  // エラー時は早期リターン
    Ok(content)
}
```

### 3.10 トレイト（Trait）

他言語のインターフェースに近い概念。

```rust
trait Summary {
    fn summarize(&self) -> String;
}

struct Article {
    title: String,
    body: String,
}

impl Summary for Article {
    fn summarize(&self) -> String {
        format!("{}: {}...", self.title, &self.body[..20])
    }
}
```

---

## 4. 依存関係の管理

### 4.1 クレート（ライブラリ）の追加

```bash
cargo add serde            # Cargo.toml に依存を追加
cargo add serde --features derive  # フィーチャー付き
cargo add tokio --features full    # 非同期ランタイム
```

または `Cargo.toml` を直接編集:

```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
rand = "0.8"
```

### 4.2 よく使われるクレート

| クレート | 用途 |
|---|---|
| `serde` / `serde_json` | シリアライズ / JSON |
| `tokio` | 非同期ランタイム |
| `reqwest` | HTTP クライアント |
| `clap` | CLI 引数パーサー |
| `anyhow` / `thiserror` | エラーハンドリング |
| `log` / `tracing` | ロギング |
| `rand` | 乱数生成 |

---

## 5. テスト

```rust
// src/lib.rs や各モジュール内
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
    }

    #[test]
    #[should_panic]
    fn test_panic() {
        panic!("this should panic");
    }
}
```

```bash
cargo test              # 全テスト実行
cargo test test_add     # 名前でフィルタ
cargo test -- --nocapture  # println の出力を表示
```

---

## 6. 開発ツール

### 6.1 フォーマッタとリンター

```bash
rustup component add rustfmt clippy

cargo fmt       # コードフォーマット
cargo clippy    # リント（より良い書き方の提案）
```

### 6.2 ドキュメント生成

```rust
/// 2つの数値を加算する。
///
/// # Examples
///
/// ```
/// let result = my_crate::add(2, 3);
/// assert_eq!(result, 5);
/// ```
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

```bash
cargo doc --open    # ドキュメントを生成してブラウザで開く
```

---

## 7. 学習リソース

| リソース | URL | 内容 |
|---|---|---|
| The Rust Programming Language (通称 "The Book") | https://doc.rust-lang.org/book/ | 公式の入門書。まずこれを読む |
| Rust by Example | https://doc.rust-lang.org/rust-by-example/ | 例題ベースで学ぶ |
| Rustlings | https://github.com/rust-lang/rustlings | 小さな演習問題で学ぶ |
| Rust Playground | https://play.rust-lang.org/ | ブラウザ上で Rust を試せる |
| std ドキュメント | https://doc.rust-lang.org/std/ | 標準ライブラリリファレンス |

### 推奨する学習順序

1. **Rust Playground** でブラウザ上から触ってみる
2. **The Book** の 1〜10 章を読む（所有権・構造体・列挙型・エラーハンドリングまで）
3. **Rustlings** の演習で手を動かして定着させる
4. 小さなプロジェクト（CLI ツールなど）を自作する
5. **The Book** の残り（ジェネリクス、ライフタイム、クロージャ、並行性など）を読む
6. 実践的なクレートを使ったアプリケーションを作る
