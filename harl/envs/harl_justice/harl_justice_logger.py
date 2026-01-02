from thesis_rl.HARL.harl.common.base_logger import BaseLogger


class HarlJusticeLogger(BaseLogger):
    def get_task_name(self):
        return "harl_justice"