import os
import random
import string
import uuid
import argparse
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# 常见验证码尺寸
COMMON_SIZES = [
    (100, 40), (120, 40), (130, 50),
    (140, 50), (160, 60)
]

def load_fonts(font_dir):
    fonts = []
    for f in os.listdir(font_dir):
        if f.lower().endswith(('.ttf', '.otf')):
            fonts.append(os.path.join(font_dir, f))
    if not fonts:
        raise Exception("fonts 目录下没有字体文件")
    return fonts

def random_color(start=0, end=255):
    return (
        random.randint(start, end),
        random.randint(start, end),
        random.randint(start, end)
    )

def generate_captcha(fonts, charset, length):
    width, height = random.choice(COMMON_SIZES)

    image = Image.new('RGB', (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(image)

    text = ''.join(random.choice(charset) for _ in range(length))

    # 动态字体大小
    base_font_size = int(height * 0.7)

    char_width = width // (length + 1)

    for i, char in enumerate(text):
        font_path = random.choice(fonts)
        font_size = base_font_size + random.randint(-4, 4)

        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()

        # 蓝色系
        color = (
            random.randint(0, 60),
            random.randint(80, 160),
            random.randint(150, 255)
        )

        x = i * char_width + random.randint(5, 10)
        y = random.randint(0, height // 4)

        # 单字符图层
        char_img = Image.new('RGBA', (char_width, height), (0, 0, 0, 0))
        char_draw = ImageDraw.Draw(char_img)
        char_draw.text((0, 0), char, font=font, fill=color)

        # 旋转
        angle = random.randint(-30, 30)
        char_img = char_img.rotate(angle, expand=1)

        image.paste(char_img, (x, y), char_img)

    # 干扰线
    for _ in range(random.randint(1, 3)):
        draw.line(
            (
                random.randint(0, width),
                random.randint(0, height),
                random.randint(0, width),
                random.randint(0, height)
            ),
            fill=random_color(120, 200),
            width=1
        )

    # 噪点
    for _ in range(random.randint(80, 150)):
        draw.point(
            (random.randint(0, width), random.randint(0, height)),
            fill=random_color(150, 255)
        )

    # 仿射变换（轻微扭曲）
    image = image.transform(
        image.size,
        Image.AFFINE,
        (
            1,
            random.uniform(-0.3, 0.3),
            0,
            random.uniform(-0.2, 0.2),
            1,
            0
        ),
        Image.BILINEAR
    )

    # 模糊
    if random.random() < 0.5:
        image = image.filter(ImageFilter.SMOOTH)

    return image, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--charset", type=str,
                        default=string.ascii_letters + string.digits)
    parser.add_argument("--length", type=int, default=4)
    parser.add_argument("--font_dir", type=str, default="fonts")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fonts = load_fonts(args.font_dir)

    for i in range(args.num_samples):
        img, label = generate_captcha(fonts, args.charset, args.length)

        filename = f"{label}_{uuid.uuid4().hex}.png"
        path = os.path.join(args.output_dir, filename)

        img.save(path)

        if i % 100 == 0:
            print(f"已生成 {i} 张")

    print("生成完成 ✅")


if __name__ == "__main__":
    main()