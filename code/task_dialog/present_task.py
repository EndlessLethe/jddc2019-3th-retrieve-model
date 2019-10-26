import re
import random

from code.task_dialog.tools import ch_count



def intent_update(msg, dialog_status):
    if re.compile("赠品").search(msg):
        dialog_status.intent = "present"
    return dialog_status


def present_handle(msg, dialog_status):
    dialog_status.intent = None
    return "赠品赠完即止，就是没有了的，以订单显示为准的哦"
