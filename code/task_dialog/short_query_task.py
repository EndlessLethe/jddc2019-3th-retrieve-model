#-*- coding: utf-8 -*-


import re
import random

from code.task_dialog.tools import ch_count

ch_pattern = re.compile(r"[\u4e00-\u9fa5]+")
match_pattern = re.compile(r"好|哦|嗯|哈|麻烦")
not_match_pattern = re.compile(r"吗|\?|？|多|哪|怎|什么|啥|退|发票")
bracket_pattern = re.compile(r"\[.*\]")


def intent_update(msg, dialog_status):
    # msg = bracket_pattern.sub("括", msg)
    if ch_count(msg) <= 4 and ch_count(msg) >=1 and len(msg) < 8 and match_pattern.search(msg):
        if not ch_pattern.search(msg) or not not_match_pattern.search(msg):
                dialog_status.intent = "short_query"
    return dialog_status


def short_query_handle(msg, dialog_status):
    responses = [
        "亲爱哒，很高兴遇到您这么善解人意的客户，请问还有其他可以帮到您的吗?"]
    response = random.sample(responses, 1)[0]
    return response
