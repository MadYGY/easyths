import os
import uuid
import random
import argparse
import string
from PIL import Image, ImageDraw, ImageFont


def get_args():
    parser = argparse.ArgumentParser(description="同花顺风格多尺寸验证码数据集生成器")
    parser.add_argument("--num_samples", type=int, default=10, help="生成样本数量")
    parser.add_argument("--code", type=str,
                        default=string.ascii_letters + string.digits,
                        help="字符集")
    parser.add_argument("--output_dir", type=str, default="./dataset", help="输出目录")
    parser.add_argument("--font_dir", type=str, default="./fonts", help="字体库目录")
    parser.add_argument("--length", type=int, default=4, help="验证码字符长度")
    # 允许传入多个尺寸，格式为 WxH
    parser.add_argument("--sizes", type=str, nargs='+',
                        default=["120x40", "150x50", "160x60", "180x70"],
                        help="可选的尺寸列表")
    return parser.parse_args()


def get_random_font(font_dir, size):
    if os.path.exists(font_dir):
        fonts = [os.path.join(font_dir, f) for f in os.listdir(font_dir)
                 if f.endswith(('.ttf', '.otf', '.TTF'))]
        if fonts:
            return ImageFont.truetype(random.choice(fonts), size)
    return ImageFont.load_default()


def generate_captcha(text, output_path, font_dir, image_size):
    width, height = image_size
    bg_color = (255, 255, 255)
    text_color = (0, 160, 233)

    image = Image.new('RGB', (width, height), bg_color)

    # 核心逻辑：根据高度动态计算字体大小 (高度的 70% 到 85%)
    base_font_size = int(height * random.uniform(0.7, 0.85))

    # 计算字符起始位置和间距
    current_x = int(width * 0.1)  # 留出 10% 的左边距
    char_step = int((width * 0.8) / len(text))  # 均匀分布字符

    for char in text:
        font = get_random_font(font_dir, base_font_size)

        # 为字符创建层，尺寸略大于字体以容纳旋转
        char_layer_size = (int(base_font_size * 1.5), int(base_font_size * 1.5))
        char_img = Image.new('RGBA', char_layer_size, (0, 0, 0, 0))
        char_draw = ImageDraw.Draw(char_img)

        # 居中绘制字符
        char_draw.text((10, 10), char, font=font, fill=text_color)

        # 随机旋转
        char_img = char_img.rotate(random.randint(-15, 15), expand=1, resample=Image.BICUBIC)

        # 垂直方向随机偏移
        y_max_offset = height - base_font_size
        y_offset = random.randint(0, max(0, y_max_offset))

        # 粘贴到主图
        image.paste(char_img, (current_x, y_offset - 10), char_img)

        # 步进横坐标，增加一点随机抖动
        current_x += char_step + random.randint(-2, 2)

    # 添加少量噪点
    draw = ImageDraw.Draw(image)
    for _ in range(int(width * height * 0.005)):  # 噪点密度随面积缩放
        draw.point((random.randint(0, width), random.randint(0, height)), fill=(210, 210, 210))

    image.save(output_path)


def main():
    args = get_args()
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

    # 解析尺寸列表
    parsed_sizes = []
    for s in args.sizes:
        w, h = map(int, s.lower().split('x'))
        parsed_sizes.append((w, h))

    print(f"开始生成！包含尺寸: {args.sizes}")

    for i in range(args.num_samples):
        label = ''.join(random.choices(args.code, k=args.length))
        file_uuid = uuid.uuid4().hex[:8]
        filename = f"{label}_{file_uuid}.png"

        # 随机选择一个尺寸
        chosen_size = random.choice(parsed_sizes)

        generate_captcha(label, os.path.join(args.output_dir, filename), args.font_dir, chosen_size)

        if (i + 1) % 100 == 0:
            print(f"进度: {i + 1}/{args.num_samples}")

    print(f"完成！图片已存至: {args.output_dir}")


if __name__ == "__main__":
    main()