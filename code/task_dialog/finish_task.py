#-*-coding:utf-8-*-


import re
import random
from code.task_dialog.tools import ch_count


finish_pattern = re.compile(
    r"谢|"
    r"没.*了|"
    r"再见|"
    r"拜拜")


def intent_update(msg,dialog_status):  
    if ch_count(msg) <= 8 and finish_pattern.search(msg):
        dialog_status.intent = "finish"
    return dialog_status


def finish_handle(sentence,dialog_status):
    if re.compile("谢谢").search(sentence):
        return "您太客气了，都是我应该做的，您看还有其他什么可以帮到您的么？"
    goodbye_sheet = [
        "您太客气了呢，这都是我应该做的呢~请问还有其他还可以帮到您的吗?妹子祝福您幸福快乐，前程锦绣，还请您点击表情栏旁边的“+”打赏我一个评价哦，感谢您对京东的支持，祝您生活愉快，再见!",
        "亲爱哒，感谢您对京东的支持，祝您生活愉快~",
        "妹子祝福您幸福快乐，前程锦绣，还请您点击表情栏旁边的“+”打赏我一个评价哦"]
    response = random.sample(goodbye_sheet, 1)[0]
    return response
