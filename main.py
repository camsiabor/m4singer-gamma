# This is a sample Python script.
import re


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    files = ['checkpoints/m4singer_hifigan/model_ckpt_steps_1970000.ckpt']

    files = sorted(
        files,
        key=lambda x: int(re.findall(r'model_ckpt_steps_(\d+).ckpt', x)[0])
    )
    print(files)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
