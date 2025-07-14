import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.applications import MobileNet

# تابع برای ساخت مدل NIMA بر پایه MobileNet
def build_nima_model(input_shape=(224, 224, 3)):
    """
    ایجاد مدل NIMA با پایه MobileNet و افزودن لایه‌های نهایی برای امتیازدهی.
    """
    base_model = MobileNet(input_shape=input_shape, include_top=False, pooling='avg', weights='imagenet')
    
    # لایه‌های سفارشی بالای مدل پایه
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x) # 10 خروجی برای امتیاز 1 تا 10

    return Model(base_model.input, x)

# تابع اصلی برای اجرا
def main():
    # --- تنظیمات ---
    # نام فایل وزن‌های مدل که دانلود خواهی کرد
    weights_file = 'mobilenet_aesthetic_weights.h5'
    # نام تصویر ورودی
    image_path = 'your_image.jpg' # <<-- نام عکس خودت را اینجا بگذار

    # 1. ساخت و بارگذاری مدل
    print("در حال بارگذاری مدل...")
    model = build_nima_model()
    try:
        model.load_weights(weights_file)
    except IOError:
        print(f"خطا: فایل وزن‌های مدل '{weights_file}' پیدا نشد.")
        print("لطفاً مطمئن شوید که فایل را دانلود کرده و در پوشه صحیح قرار داده‌اید.")
        return

    # 2. بارگذاری و پیش‌پردازش تصویر
    print(f"در حال پردازش تصویر: {image_path}...")
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"خطا: فایل تصویر '{image_path}' پیدا نشد.")
        return

    image = image.resize((224, 224))
    image_array = np.array(image)
    
    # اگر تصویر کانال آلفا (شفافیت) داشت، آن را حذف کن
    if image_array.shape[2] == 4:
        image_array = image_array[..., :3]

    # نرمال‌سازی و افزودن بعد بچ (batch dimension)
    image_array = image_array / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    # 3. پیش‌بینی امتیاز
    print("در حال محاسبه امتیاز زیبایی...")
    score_distribution = model.predict(image_batch)

    # 4. محاسبه و نمایش امتیاز نهایی
    # امتیاز نهایی میانگین وزنی امتیازهای ۱ تا ۱۰ است
    mean_score = np.sum(score_distribution * np.arange(1, 11))
    
    print("\n" + "="*30)
    print(f"⭐ امتیاز زیبایی تصویر: {mean_score:.2f} از 10")
    print("="*30)

if __name__ == '__main__':
    main()