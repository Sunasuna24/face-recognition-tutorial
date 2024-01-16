from pathlib import Path            # スタンダードなライブラリ
import face_recognition             # サードパーティーのライブラリ
import pickle                       # スタンダードなライブラリ
from collections import Counter     # 最も一致するモデルの投票に用いる
from PIL import Image, ImageDraw    # PILからImageとImageDrawモジュールをインポート
import argparse

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")   # デフォルトの向き先
BOUNDING_BOX_COLOR = "blue"                             # bouding boxの色（HTMLで一般に用いられる色名を指定）
TEXT_COLOR = "white"                                    # 注釈の文字色（HTMLで一般に用いられる色名を指定）

Path("training").mkdir(exist_ok=True)   # ディレクトリを作成する
Path("output").mkdir(exist_ok=True)     # ディレクトリを作成する
Path("validation").mkdir(exist_ok=True) # ディレクトリを作成する

parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument("--validate", action="store_true", help="Validate trained model")
parser.add_argument("--test", action="store_true", help="Test the model with an unknown image")
parser.add_argument("-m", action="store", default="hog", choices=["hog", "cnn"], help="Which model to use for training: hog (CPU), cnn (GPU)")
parser.add_argument("-f", action="store", help="Path to an image with an unknown face")

args = parser.parse_args()


"""encode_known_faces関数
トレーニング用の画像を読み込んで、画像から顔を検出、それぞれの画像についての名前-エンコーディングのdictiroaryを作成する関数。

引数1: string、モデルの方。HOG
引数2: エンコーディングした情報を格納するパス。DEFAULT_ENCODINGS_PATH
"""
def encode_known_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    names = []
    encodings = []

    # /trainingディレクトリを見る
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)  # /trainingディレクトリ内の画像ファイルを読み込む

        face_locations = face_recognition.face_locations(image, model=model)    # 画像から顔を検出する
        face_encodings = face_recognition.face_encodings(image, face_locations) # 検出した顔のエンコーディングを取得する
        """face_encodings変数の中身の説明
        簡単に言うと、数字の配列。顔の特徴を表す。
        """

        # face_encodingsというリストにディレクトリ名(=ファイル名？)とエンコーディングを追加する。
        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    # pickleを使って、ファイル名とエンコード情報を保存して、encodings.pklというファイルを作成する
    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

"""recognize_faces関数
ラベル付けされていない画像の顔を識別することができる関数。

引数1: string、識別したい未ラベルの画像のパス
引数2: string、モデルの方。HOG
引数3: エンコーディングした情報を格納するパス。DEFAULT_ENCODINGS_PATH
"""
def recognize_faces(image_location: str, model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    # ファイルに保存したエンコーディングを開ける
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)                              # 入力した未ラベルの画像を読み込ませる
    input_face_locations = face_recognition.face_locations(input_image, model=model)            # 未ラベルの画像の顔を検出する
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)   # 未ラベルの画像の顔のエンコーディングを取得する

    # 検出したモデルを描画するため
    pillow_image = Image.fromarray(input_image) # 与えた未ラベルの画像のPillow imageオブジェクトを生成する
    draw = ImageDraw.Draw(pillow_image)         # それを元に、PillowDrawオブジェクトを生成し、検出した顔の周りにbounding boxを描画する。

    # 入力した未ラベルの画像を（for文を通して）過去に作成したエンコーディングの塊に晒して比較する。
    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        # 識別がどのモデルとも一致しない場合は"Unknown"という名前を付ける
        if not name:
            name = "Unknown"

        # bouding_boxと一緒に検出したモデルをディスプレイに出力する
        _display_face(draw, bounding_box, name)

    del draw
    pillow_image.show() # 画像を表示する

"""_recognize_face関数
未ラベルの画像とモデルの両方のエンコーディングを比較して、（もしあれば）一致するモデルを返す

引数1: 未ラベルの画像のエンコーディング
引数2: モデルのエンコーディング
"""
def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)

    # Counterを用いて、よりマッチするモデルを選択する
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]

"""_display_face関数
検出した顔にbounding boxを描画し、そのbouding boxに検出したモデルの注釈を入れてくれる関数。
どのモデルとも一致しなかったら、Unknownと注釈する。
"""
def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)
    draw.rectangle(((text_left, text_top), (text_right, text_bottom)), fill="blue", outline="blue")
    draw.text((text_left, text_top), name, fill="white")

"""validate関数
大まかに言うと、正解が分かっている、モデルが未学習のデータを学習モデルを与えることで、モデルの性能をテストすること。
/validationディレクトリにある全ての画像を上記で作成したrecognize_faces関数に入れて検証させる。
"""
def validate(model: str = "hog"):
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(image_location=str(filepath.absolute()), model=model)

if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate(model=args.m)
    if args.test:
        recognize_faces(image_location=args.f, model=args.m)