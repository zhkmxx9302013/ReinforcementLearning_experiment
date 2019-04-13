import tensorflow as tf
class ResultLogger:
    def __init__(self, writer):
        """

        :param writer: TensorboardX writer
        """
        self.writer = writer
        self.score = []
        self.mean = []
        self.episode = 0
        pass

    def log_result(self, total_reward, winnum, drawnum, episode):
        """

        :param total_reward:
        :param winnum:
        :param drawnum:
        :param episode:
        :return:
        """
        self.episode=episode
        self.score.append(total_reward)
        self.writer.add_scalar('total_reward', total_reward, episode)
        mean_reward = sum(self.score[-100:]) / 100
        self.mean.append(mean_reward)
        self.writer.add_scalar('mean_reward', mean_reward, episode)

        if episode % 100:
            self.writer.add_scalar('win_rate', winnum / 100, episode)
            self.writer.add_scalar('draw_rate', drawnum / 100, episode)
        pass

    def log_parameter(self, info_dict=None):
        """
        Log hyper parameter.
        :param info_dict:
        :return:
        """
        if info_dict and type(info_dict) == dict:
            for (k, v) in info_dict.items():
                self.writer.add_scalar(k, v, self.episode)

        pass

    def log_gaes(self, gae, episode):
        self.writer.add_scalar('GAE', gae, episode)

    def log_info(self, info_dict=None, episode=0):
        """
        Log hyper parameter.
        :param info_dict:
        :return:
        """
        if info_dict and type(info_dict) == dict:
            for (k, v) in info_dict.items():
                self.writer.add_scalar(k, v, episode)

        pass