from pypinyin import lazy_pinyin, Style
import re

def process_initials_finals(x: str, y: str) -> str:
    # 如果x非空
    if x:
        # 且y非空
        if y:
            return ' '.join((x, y))
        else:
            return x
    # 如果x为空，则y必定非空，直接返回y（当声母不存在的情况下syx）
    else:
        return y


def generate_pinyin_seq(string):
    """
    将一句话，转换成拼音序列
    “小猪佩奇” --> “x iao zh u p ei q i”
    :param string: 传入一个句子
    :return: 返回拼音序列
    """

    #(syx)x为得到的字的拼音首字母，y为得到的剩余所有，指定strict为false去掉y等无声韵母规则
    pinyin_english_list = []
    my_re = re.compile(r'[\u4e00-\u9fa5]')
    res = re.findall(my_re, string)
    for word in string:
        if word in res:
            initials_list = lazy_pinyin(res, style=Style.INITIALS, strict=False)
            finals_list = lazy_pinyin(res, style=Style.FINALS, strict=False)
            zipitem = list(zip(initials_list,finals_list))
            for i in zipitem:
                pinyin_english_list.append(i)
        else:
            pinyin_english_list.append(word)
    # return ' '.join([' '.join((x, y)) for x, y in zip(initials_list, finals_list)])
    print(pinyin_english_list)
    return ' '.join([process_initials_finals(x, y) for x, y in zip(initials_list, finals_list)])


if __name__ == '__main__':
    print(generate_pinyin_seq('小猪 pig 佩奇'))
