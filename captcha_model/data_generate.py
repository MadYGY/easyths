import os
import random
import string
import uuid
import argparse
from PIL import Image, ImageDraw, ImageFont


# 同花顺验证码固定尺寸（精确匹配真实样本: 84x38）
CAPTCHA_WIDTH = 84
CAPTCHA_HEIGHT = 38

# 同花顺验证码的主文字色（通过像素统计分析得到）
# 核心色: RGB(0, 160, 233)，占绝大多数
TEXT_COLOR = (0, 160, 233)

# 顶部边线颜色（真实验证码 y=0 行固定有一条淡蓝色边线）
BORDER_TOP_COLORS = [
    (219, 233, 242),  # 主要（61/84 像素）
    (218, 232, 241),  # 次要（12/84 像素）
    (220, 234, 243),  # 次要（11/84 像素）
]
BORDER_TOP_WEIGHTS = [61, 12, 11]


def load_fonts(font_dir):
    """加载字体文件"""
    fonts = []
    for f in os.listdir(font_dir):
        if f.lower().endswith(('.ttf', '.otf')):
            fonts.append(os.path.join(font_dir, f))
    if not fonts:
        raise Exception("fonts 目录下没有字体文件")
    return fonts


def draw_top_border(draw, width):
    """绘制顶部淡蓝色边线（同花顺验证码固定特征）"""
    for x in range(width):
        color = random.choices(BORDER_TOP_COLORS, weights=BORDER_TOP_WEIGHTS, k=1)[0]
        draw.point((x, 0), fill=color)


def generate_captcha(fonts, charset, length):
    """
    生成仿同花顺风格验证码

    通过逐像素分析真实同花顺验证码得到的精确特征：
    - 尺寸: 84x38 像素
    - 背景: 纯白 RGB(255,255,255)
    - 顶部 y=0 行: 淡蓝色边线 RGB(219,233,242)
    - 文字色: 天蓝 RGB(0,160,233)，同一张图颜色一致
    - 文字区域: 约 y=13 至 y=29（垂直居中偏下）
    - 字符 x 范围: 约 x=2 至 x=79
    - 无旋转、无仿射变换、无干扰线、无人为噪点
    - 仅有字体抗锯齿产生的自然过渡色
    """
    width, height = CAPTCHA_WIDTH, CAPTCHA_HEIGHT

    # 纯白背景
    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 绘制顶部边线
    draw_top_border(draw, width)

    # 生成随机文本
    text = ''.join(random.choice(charset) for _ in range(length))

    # 字体大小（真实验证码文字较粗，字体约18-20pt）
    base_font_size = 19

    # 字符布局参数（基于真实样本间隙分析）
    # 真实样本字符间隙位置: ~(20,23), ~(31,43), ~(52,63)
    # 即4个字符分别在约 x=2~20, x=23~31, x=43~52, x=63~79
    # 简化：每个字符区域约20像素宽，起始位置约 x=2, 23, 43, 63
    char_start_positions = [3, 23, 43, 63]

    for i, char in enumerate(text):
        font_path = random.choice(fonts)
        font_size = base_font_size + random.randint(-1, 1)

        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        # 获取字符尺寸来居中
        bbox = font.getbbox(char)
        char_w = bbox[2] - bbox[0]
        char_h = bbox[3] - bbox[1]

        # x坐标：基于固定位置 + 轻微随机偏移
        if i < len(char_start_positions):
            x = char_start_positions[i] + random.randint(-1, 2)
        else:
            x = 3 + i * 20 + random.randint(-1, 2)

        # y坐标：居中在画面中（文字区域约 y=13~29，中心约 y=21）
        # 真实验证码中文字垂直居中，有轻微 y 抖动
        y_center = (height - char_h) // 2 + 2  # 略偏下
        y = y_center + random.randint(-2, 2)

        # 直接绘制，不旋转（同花顺验证码无旋转）
        draw.text((x, y), char, font=font, fill=TEXT_COLOR)

    # 不添加任何干扰线或噪点（真实验证码中没有人为干扰）

    return image, text


def main():
    parser = argparse.ArgumentParser(description="生成仿同花顺验证码数据集")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="生成样本数量")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="输出目录")
    parser.add_argument("--charset", type=str,
                        default=string.ascii_letters + string.digits,
                        help="字符集")
    parser.add_argument("--length", type=int, default=4,
                        help="验证码字符数")
    parser.add_argument("--font_dir", type=str, default="fonts",
                        help="字体目录")

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

    print(f"生成完成 ✅ 共 {args.num_samples} 张，保存在 {args.output_dir}/")


if __name__ == "__main__":
    main()
