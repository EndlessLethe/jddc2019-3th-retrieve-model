#!/usr/bin/python3
# -*- coding: utf-8 -*-
from .task_dialog.task_core import TaskCore


class DialogStatus(object):

    def __init__(self):
        self.intent = None

        # Universal slots
        self.ware_id = None
        self.order_id = None

        # Special slots
        self.start_flag = None
        self.sale_return_intent = None
        self.invoice_intent = None
        self.query_intent = None
        self.order_related = None

        # unbind
        self.unbind_flag = None
        self.unbind_identify = None
        self.unbind_phone = None
        self.unbind_new_phone = None
        self.unbind_success = None

        # price protect
        self.price_protect_success = None

        # dialog context
        self.context = []


class DialogManagement(object):
    dialog_status = DialogStatus()

    @classmethod
    def init_status(cls):
        cls.dialog_status = DialogStatus()

    @classmethod
    def process_dialog(cls, msg):
        task_response, cls.dialog_status = TaskCore.task_handle(msg, cls.dialog_status)
        if task_response:
            response = task_response
        else:
            response = ' '
        return response


def process_single_question_in_session(question):
    DialogManagement.dialog_status.context.append(question)
    response = DialogManagement.process_dialog(question)
    DialogManagement.dialog_status.context.append(response)
    return response


def process_question_list(id_list, question_list):
    flag_list = []
    response_list = []
    if len(id_list) > 0:
        cur_id = id_list[0]
        DialogManagement.init_status()
    else:
        return [], []
    for id, question in zip(id_list, question_list):
        if id != cur_id:
            DialogManagement.init_status()
            cur_id = id
        response = process_single_question_in_session(question)
        flag_list.append(False if response == ' ' else True)
        response_list.append(None if response == ' ' else response)

    return flag_list, response_list