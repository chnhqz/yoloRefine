"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.

PyTorch 移植：该代码是将原始扩散模型实现的TensorFlow（TF）版本移植到PyTorch。这意味着代码已重新编写，以使用PyTorch的深度学习框架，而不是TensorFlow。
文档（文档字符串）：代码现在包含文档字符串。文档字符串是添加到代码中的注释，用于描述函数、类和模块的目的和用法。这使其他开发人员更容易理解和使用代码。
DDIM 抽样：DDIM代表“去噪扩散隐式模型”。这是用于生成建模的技术，特别用于图像生成。代码似乎包括DDIM抽样，这是从扩散模型中的概率模型中抽样的方法。
Beta 计划：代码包括一组Beta计划。在扩散模型的背景下，Beta通常是控制模型去噪速率的超参数。Beta计划是函数或值，确定在模型训练过程中随着时间的推移如何改变Beta。这组Beta计划可能提供了在训练期间不同的Beta变化策略。
总的来说，该代码似乎是Jonathan Ho扩散模型实现的定制版本。它包括PyTorch的适应、改进的文档、DDIM抽样功能以及各种Beta计划选项，使其更加灵活和用户友好，适用于生成建模任务。
"""

import enum
import math

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.

    这段代码定义了一个函数get_named_beta_schedule，用于获取预定义的beta（β）调度（schedule）。
    在扩散模型中，β是一个控制去噪过程速度的超参数，不同的β调度可以影响模型的训练行为。下面是这个函数的解释：
    get_named_beta_schedule(schedule_name, num_diffusion_timesteps): 这是一个函数，用于获取指定名称的β调度。
        schedule_name: 要获取的β调度的名称，可以是"linear"（线性）或"cosine"（余弦）。
        num_diffusion_timesteps: 表示扩散模型中的时间步数量。这个参数用于根据模型的不同需求确定β的调度。
    该函数的实现分为两个部分，根据不同的schedule_name返回不同的β调度：
        如果schedule_name为"linear"，则会返回一个线性β调度。这个β调度会从一个很小的初始值（beta_start）线性增加到一个稍大的终止值（beta_end）。
    这个调度被设计为在给定的num_diffusion_timesteps时间步内逐渐增加β的值，以实现模型的去噪过程。
        如果schedule_name为"cosine"，则会返回一个余弦β调度。这个调度的设计基于余弦函数，通过调整余弦函数的参数，可以获得不同的β值。这种调度可能适用于一些特定的应用场景。
        如果schedule_name不是"linear"或"cosine"，则会引发一个NotImplementedError异常，指示该β调度名称未知。
    这个函数允许用户根据需要选择不同的β调度，以在扩散模型中控制去噪过程的速度和行为。
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    这段代码定义了一个函数betas_for_alpha_bar，用于创建一个离散化的β（beta）调度，这个调度是根据给定的alpha_t_bar函数定义的，
    该函数表示从时间t = [0, 1]的某一部分开始的(1-beta)的累积乘积。以下是这个函数的解释：
    betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999): 这是一个函数，用于生成β调度。num_diffusion_timesteps: 表示要生成的β的数量，通常与扩散模型中的时间步数量对应。
    alpha_bar: 一个函数（lambda），它接受一个介于0到1之间的参数t，并生成从扩散过程开始到t时刻的(1-beta)的累积乘积。这个函数描述了扩散过程中的(1-beta)的积累情况。
    max_beta: 最大的β值，通常小于1，用于防止奇异性（singularities）。
    该函数的实现主要是通过循环生成一系列的β值，以便在整个扩散过程中控制β的变化。具体步骤如下：
        使用一个循环，迭代从0到num_diffusion_timesteps-1。
        对于每次迭代，计算t1和t2，这些值用于确定当前时间步和下一个时间步的时间点。这样可以根据alpha_bar函数的输出来计算当前时间步的β值。
        计算β值时，使用了如下公式：1 - alpha_bar(t2) / alpha_bar(t1)。这个公式表示当前时间步到下一个时间步之间(1-beta)的累积乘积，然后通过减去这个值来得到β。
    然后，将计算得到的β值与max_beta相比较，以确保不会超过指定的最大值。
        将生成的β值添加到betas列表中。
        最后，将生成的β值列表转换为NumPy数组，并返回。
    这个函数的主要目的是根据给定的alpha_t_bar函数，生成一个适用于扩散模型的β调度，以控制模型在训练期间的去噪过程。这种调度可以通过提供不同的alpha_bar函数来适应不同的去噪行为。
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).

    这段代码是一段文档字符串（docstring），用于提供有关训练和采样扩散模型所需的一些信息和参数的说明。它提供了对代码模块的概述和参数的解释。以下是这段文档字符串的解释：
    Utilities for training and sampling diffusion models：这是文档字符串的标题，说明了这段代码是用于扩散模型的训练和采样的一些工具。
    Ported directly from here, and then adapted over time to further experimentation：这是一段注释，说明该代码是从指定的GitHub链接中直接移植而来，然后随着时间的推移进行了一些适应和实验性的改进。
    Parameters：下面是关于函数参数的说明：
        betas: 一个包含β值的1-D NumPy数组，表示每个扩散时间步的β值。这些β值从T（初始时间）开始逐渐减小到1。
        model_mean_type: 一个枚举类型（ModelMeanType），用于确定模型的输出是什么类型。这个参数可能用于指定模型的输出是均值（mean）的哪种类型。
        model_var_type: 一个枚举类型（ModelVarType），用于确定模型如何输出方差（variance）。这个参数可能用于指定模型如何输出方差信息。
        loss_type: 一个枚举类型（LossType），用于确定要使用的损失函数类型。这个参数可能用于指定在模型训练中使用的损失函数。
        rescale_timesteps: 一个布尔值，如果设置为True，将浮点数时间步传递给模型，以使它们始终按照原始论文中的方式进行缩放（从0到1000）。这个参数可能用于确保时间步的一致性和缩放。
    这个文档字符串的目的是提供有关函数的输入参数和用途的说明，以便用户或其他开发者可以更好地理解和使用相关的工具函数。

    """

    def __init__(
        self,
        *,
        betas,  # 包含bata值的Numpy数组 表示每个扩散时间步的beta值
        model_mean_type,  # 一个枚举类型，用于确定模型的输出是什么类型
        model_var_type,  # 一个枚举类型， 用于确定模型如何输出方差
        loss_type,  # 一个枚举类型， 用于确定要使用的损失函数类型
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas

        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])  # 时间步，与beta长度一致

        alphas = 1.0 - betas  # alpha , 跟论文中定义的一摸一样
        self.alphas_cumprod = np.cumprod(alphas, axis=0)  # alphas 的累乘 (0, n)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[
                                                  :-1])  # alphas 的累乘 (1, n) 输出一个拼接的numpy数组[1.0, alphas_cumprod_prev]
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:],
                                             0.0)  # alphas 的累乘 (0, n-1) 输出一个拼接的numpy数组[alphas_cumprod_next, 0]
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # 计算q(x_t | x_{t-1}) 和其他扩散
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        '''
        这段代码执行了一系列计算，用于计算在给定条件下的后验分布 q(x_{t-1} | x_t, x_0) 的参数，包括均值和方差。这些计算似乎与扩散模型中的概率分布相关，以下是这些计算的解释：
        1. self.posterior_variance: 这个变量用于计算后验分布的方差。具体计算方式是根据扩散模型中的参数betas、
        前一时间步的alphas_cumprod_prev、和当前时间步的alphas_cumprod来计算。它表示后验分布的方差。
        2. self.posterior_log_variance_clipped: 这个变量用于计算截断后的后验分布方差的对数。这是因为在扩散链的开始阶段，后验方差为0，
        所以对数的计算需要进行截断以避免出现无穷大的情况。该变量包含了一个经过截断的后验方差数组。
        3. self.posterior_mean_coef1 和 self.posterior_mean_coef2: 这两个变量用于计算后验分布的均值的系数。具体计算方式涉及了betas、
        alphas_cumprod_prev、以及alphas。它们是后验均值的组成部分，用于确定均值的不同系数。
        这些计算通常用于扩散模型中，以便在采样过程中根据给定的条件计算后验分布的参数，从而生成合成数据或进行其他相关任务。这些参数对于理解和控制模型的行为至关重要。
        '''
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        这是一个Python函数，它计算给定初始输入 x_start(即x_0 clean image) 和扩散步数 t 下，分布 q(x_t | x_0) 的均值、方差和对数方差。
        这通常在扩散模型（可能是随机过程或随机微分方程）的上下文中使用，其中 q(x_t | x_0) 表示在时间 t 时刻给定初始条件 x_0 下，
        随机变量 x_t 的条件概率分布。
        具体地说，这个函数的三个返回值是：
        mean（均值）：这是一个与 x_start 具有相同形状的张量，表示在时间 t 时刻给定初始条件 x_0 下，随机变量 x_t 的均值。
        均值计算的方式是将 x_start 乘以一个关于扩散步数 t 的系列 sqrt_alphas_cumprod。

        variance（方差）：这也是一个与 x_start(即x_0 clean image) 具有相同形状的张量，表示在时间 t 时刻给定初始条件 x_0 下，随机变量 x_t 的方差。
        方差计算的方式是使用与扩散步数 t 相关的系列 1.0 - self.alphas_cumprod。

        log_variance（对数方差）：这也是一个与 x_start(即x_0 clean image) 具有相同形状的张量，表示在时间 t 时刻给定初始条件 x_0 下，随机变量 x_t 的对数方差。
        对数方差计算的方式是使用与扩散步数 t 相关的系列 self.log_one_minus_alphas_cumprod。
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    # 这个是关键函数，对应于 noise predictor
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        这是一个Python函数，用于模拟在给定初始数据 x_start(即x_0 clean image) 和扩散步数 t 下，从条件概率分布 q(x_t | x_0) 中采样数据，
        从而生成一个具有噪声的版本的 x_start。

        函数的操作步骤如下：
            如果没有提供 noise 参数，就会生成一个与 x_start 具有相同形状的随机噪声（以正态分布随机数的形式）。这个随机噪声用于引入扰动。
        确保随机噪声的形状与 x_start 相同。
            计算并返回一个 "noisy" 版本的 x_start，这是通过以下两部分组成的：
                第一部分是 x_start 乘以一个与扩散步数 t 相关的系列 sqrt_alphas_cumprod。
                第二部分是随机噪声乘以一个与扩散步数 t 相关的系列 sqrt_one_minus_alphas_cumprod。
        这个过程可以被视为在每个扩散步骤中，首先对原始数据 x_start 应用一个变换，然后再添加一些噪声以生成新的数据点。
        这是在模拟扩散过程中，逐步引入噪声以改变数据分布的一种方式。这通常用于生成具有不同程度噪声的数据，以用于训练或测试各种机器学习或统计模型。
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        这段代码定义了一个函数q_posterior_mean_variance，用于计算扩散模型中后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差。以下是该函数的解释：
        q_posterior_mean_variance(self, x_start, x_t, t): 这是一个函数，用于计算后验分布的均值和方差。
            x_start: 表示时间步t=0时刻的数据。
            x_t: 表示时间步t时刻的数据。
            t: 表示时间步t。
        该函数执行以下步骤：
            首先，检查输入的数据形状是否一致，确保x_start和x_t的形状相同。
            然后，计算后验分布的均值。均值计算使用了以下两部分：
            第一部分由self.posterior_mean_coef1（之前计算的系数1）与x_start相乘，这部分对应于时间步t-1时刻的数据。
            第二部分由self.posterior_mean_coef2（之前计算的系数2）与x_t相乘，这部分对应于时间步t时刻的数据。
            两部分均值通过这些系数的权重相加得到。
            计算后验分布的方差，方差由self.posterior_variance中相应的值提取出来。这个方差表示在给定条件下的后验方差。
            计算截断后的后验分布方差的对数，使用self.posterior_log_variance_clipped中的相应值。这个截断后的对数方差是为了避免在方差为0的情况下出现无穷大的问题。
            最后，确保计算得到的均值、方差和对数方差的形状与输入的x_start相同，并将它们作为结果返回。
        这个函数用于计算在扩散模型中，根据给定条件，计算时间步t-1时刻的后验均值和方差，以帮助生成合成数据或进行其他相关任务。这些参数对于理解和控制模型的行为至关重要。

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.

        这段代码定义了一个函数p_mean_variance，该函数用于将模型应用于数据 x，以获取 p(x_{t-1} | x_t) 的均值和方差，
        同时还返回初始 x_0 的预测值。以下是该函数的解释：
        p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        这是一个函数，用于计算 p(x_{t-1} | x_t) 的均值和方差。
        model: 表示模型，该模型接受一个信号和一批时间步作为输入。
        x: 表示在时间步t时刻的数据，它是一个形状为 [N x C x ...] 的张量。
        t: 表示一维张量，包含时间步的信息。
        clip_denoised: 一个布尔值，如果设置为True，将对去噪后的信号进行截断，使其位于 [-1, 1] 范围内。
        denoised_fn: 如果不是None，这是一个应用于 x_start 预测值的函数，用于对其进行处理。这个函数在进行截断之前应用。
        model_kwargs: 如果不是None，这是一个字典，包含传递给模型的额外关键字参数。这可以用于条件化模型。
        该函数执行以下步骤：
            检查输入数据的形状，确保x和t的形状合法。
            使用给定的模型 model 对输入数据 x 进行处理，并获得模型输出。
            根据模型的输出和模型变量类型（model_var_type），计算模型的均值（model_mean）和方差（model_variance）。
            如果 model_var_type 是 LEARNED 或 LEARNED_RANGE，那么从模型输出中提取均值和方差。
            如果 model_var_type 是 FIXED_LARGE 或 FIXED_SMALL，根据预定义的值计算方差和对数方差。
            处理 x_start 预测值，应用 denoised_fn 函数（如果提供）来对其进行处理，然后根据 clip_denoised 参数对其进行截断。
            根据模型均值、方差和 x_start 预测值，计算后验均值和方差，这是通过调用 q_posterior_mean_variance 函数来完成的。
            最后，返回一个包含均值、方差、对数方差和预测的 x_start 的字典。
        这个函数用于在扩散模型中，根据给定条件，计算时间步t-1时刻的后验均值和方差，同时提供了初始 x_0 的预测。
        这些参数对于理解和控制模型的行为至关重要。
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        # 使用给定的模型 model 对输入数据 x 进行处理，并获得模型输出。
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            # 如果 model_var_type 是 LEARNED 或 LEARNED_RANGE，那么从模型输出中提取均值和方差。
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                # 如果 model_var_type 是 LEARNED 提取方差
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            # 如果 model_var_type 是 FIXED_LARGE 或 FIXED_SMALL，根据预定义的值计算方差和对数方差
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            # 处理 x_start 预测值，应用 denoised_fn 函数（如果提供）来对其进行处理，然后根据 clip_denoised 参数对其进行截断
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        # 从噪声eps中预测x_0
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        # 从xprev 中预测x0
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.


        这是一个Python函数，它用于从给定时间步的模型中进行采样，并返回采样的结果。
        函数接受多个参数，包括模型、当前输入 `x`、时间步索引 `t` 等。以下是该函数的主要目的和参数：
            - `model`: 用于采样的模型。
            - `x`: 当前时间步的输入张量。
            - `t`: 时间步的索引，从0开始表示第一个扩散步骤。
            - `clip_denoised`: 如果为True，将对 `x_start` 预测进行剪裁，使其范围在 [-1, 1] 之间。
            - `denoised_fn`: 如果不为None，这是一个应用于 `x_start` 预测的函数，通常用于对预测进行一些后处理操作。
            - `cond_fn`: 如果不为None，这是一个梯度函数，类似于模型，用于条件化操作。
            - `model_kwargs`: 如果不为None，这是一个包含额外关键字参数的字典，用于传递给模型。这通常用于条件化操作。
        函数的主要操作如下：
            1. 调用 `self.p_mean_variance` 函数，该函数计算给定时间步的均值和方差，以及 `x_start` 预测。
        它还接受一些其他参数，如 `clip_denoised`、`denoised_fn` 和 `model_kwargs`。
            2. 使用 `th.randn_like(x)` 生成与当前输入 `x` 相同形状的随机噪声张量。
            3. 创建一个 `nonzero_mask`，它是一个掩码，指示在 `t == 0` 时不应添加噪声。这是因为在第一个时间步（t=0）不需要添加噪声。
            4. 如果提供了 `cond_fn`，则使用 `self.condition_mean` 函数来基于条件概率梯度更新均值，将其存储在 `out["mean"]` 中。
            5. 最后，计算采样值 `sample`，这是通过将 `mean` 加上噪声并乘以 `exp(0.5 * log_variance)` 得到的，
        同时根据 `nonzero_mask` 控制是否添加噪声。
        函数返回一个包含以下两个键的字典：
            - `'sample'`: 从模型中采样得到的随机样本。
            - `'pred_xstart'`: 对 `x_0` 的预测值（通常是初始输入值的预测）。
        总之，该函数用于在给定时间步的模型中进行采样，并返回采样结果，同时考虑了条件概率梯度、均值和方差等信息。
        这在扩散模型中通常用于生成样本序列的下一个时间步。
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )  # 计算给定时间步的均值和方差，以及 x_start 预测
        noise = th.randn_like(x)  # 生成与 x 相同形状的随机噪声张量
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise  # 计算采样值
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.

        这是一个用于计算变分下界（variational lower-bound）中的一些术语的方法。
        该方法的目的是为了评估生成模型的性能，特别是在生成图像时的性能。
        1. `_vb_terms_bpd` 方法用于获取变分下界的一些术语，这些术语通常以比特（bits）为单位来表示，以便与其他研究进行比较。
        2. 方法接受以下参数：
            - `model`：生成模型，用于生成条件概率分布。
            - `x_start`：初始图像。
            - `x_t`：在时间步 t 的图像。
            - `t`：时间步骤。
            - `clip_denoised`：一个布尔值，指示是否对去噪后的图像进行裁剪。
            - `model_kwargs`：其他模型相关的参数。
        3. 首先，方法通过调用 `q_posterior_mean_variance` 方法获取真实的均值（`true_mean`）
        和对数方差（`true_log_variance_clipped`），这些值用于计算 KL 散度（Kullback-Leibler divergence）。
        4. 接下来，方法调用 `p_mean_variance` 方法来获取生成模型的均值（`mean`）和对数方差（`log_variance`），这些值用于计算 KL 散度。
        5. 使用计算得到的均值和对数方差，计算 KL 散度（`kl`）。KL 散度用于衡量两个概率分布之间的差异。
        6. 计算解码器的负对数似然（NLL），即真实图像与生成图像之间的差异。这用于度量生成模型的性能。
        7. 最后，如果是第一个时间步（t=0），则返回解码器 NLL；否则，返回 KL 散度。
        8. 最终，`output` 包含变分下界的术语，以及生成的初始图像的预测。
        这个方法的主要目的是计算一个关于生成模型性能的度量，这些度量可以用于评估模型生成的图像与真实数据之间的相似性。
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )  # 计算真实的均值 和 对数方差
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )  # 获取生成模型的均值 和 方差
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )  # 计算真实的均值 与 模型生成之间的差异
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.

        这是一个用于计算训练损失的方法，主要用于单个时间步骤。该方法接受一些输入参数，并返回一个包含损失值的字典。
            以下是该方法的关键部分解释：
                1. `model`：这是生成模型，用于生成条件概率分布。
                2. `x_start`：这是输入数据的张量，具有形状 `[N x C x ...]`，表示 `N` 个样本的初始数据。
                3. `t`：这是一个包含时间步骤索引的批次（batch），表示每个样本的当前时间步骤。
                4. `model_kwargs`：这是一个可选参数，是一个字典，包含传递给模型的额外关键字参数。这可以用于条件模型。
                5. `noise`：这是一个可选参数，表示特定的高斯噪声。如果未指定，将使用随机噪声。
                6. `x_t`：通过调用 `q_sample` 方法，使用初始数据和噪声生成的当前时间步骤的数据。
                7. `terms`：这是一个字典，用于存储损失和其他损失相关的值。
                8. 根据 `self.loss_type` 的设置，计算不同类型的损失：
                    - 如果 `self.loss_type` 是 `LossType.KL` 或 `LossType.RESCALED_KL`，则计算 KL 散度作为损失。
                    - 如果 `self.loss_type` 是 `LossType.MSE` 或 `LossType.RESCALED_MSE`，则计算均方误差（MSE）作为损失。
                9. 损失的计算基于模型输出和目标之间的差异。目标根据 `self.model_mean_type` 的设置不同，
                可以是前一个时间步骤的数据（`ModelMeanType.PREVIOUS_X`）、初始数据（`ModelMeanType.START_X`）
                或噪声（`ModelMeanType.EPSILON`）。
                10. 如果 `self.loss_type` 是 `LossType.RESCALED_KL` 或 `LossType.RESCALED_MSE`，
                则根据时间步骤的数量对损失进行了缩放。
                11. 最后，返回一个字典，包含损失值，以及其他可能与损失相关的值，如 KL 散度等。
            这个方法的主要目的是计算训练损失，以便在训练生成模型时优化模型参数。不同的损失类型和模型设置可以用于不同的训练任务。
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.


        这是一个Python函数，其目的是计算整个变分下界（variational lower-bound），以比特每维（bits-per-dim）的方式来度量。
        函数接受多个参数，包括模型 `model`、输入张量 `x_start`、是否对去噪样本进行剪裁（`clip_denoised`）、
        以及其他可能的模型参数（`model_kwargs`）。
        下面是函数的主要操作和返回值：
        1. 首先，获取输入 `x_start` 的设备（`device`）和批处理大小（`batch_size`）等信息。
        2. 然后，初始化一些空列表 `vb`、`xstart_mse` 和 `mse`，用于存储计算结果。这些列表将在后续循环中用于累积每个时间步的计算结果。
        3. 接下来，进入时间步循环，从最后一个时间步向前遍历（`list(range(self.num_timesteps))[::-1]`）。
        4. 在循环中，首先计算当前时间步的噪声（`noise`），然后使用 `self.q_sample` 函数生成 `x_t`，
        这是通过采样当前时间步的输入值（`x_start`）和添加噪声得到的。
        5. 接着，使用 `self._vb_terms_bpd` 函数计算当前时间步的变分下界（variational lower-bound）的各个术语，
        并将计算结果存储在 `out` 变量中。
        6. 将每个时间步的变分下界术语（`output`）、x_0 的均方误差（`xstart_mse`）以及噪声的均方误差（`mse`）分别添加到
         `vb`、`xstart_mse` 和 `mse` 列表中。
        7. 在循环结束后，将 `vb`、`xstart_mse` 和 `mse` 转换为张量，并将它们在时间维度上堆叠。
        8. 计算输入 `x_start` 的先验项（`prior_bpd`）。
        9. 计算总的变分下界，将其定义为各个时间步的变分下界之和，再加上先验项。
        10. 返回一个包含以下键的字典：
            - `"total_bpd"`: 每批元素的总变分下界（以比特每维度度量）。
            - `"prior_bpd"`: 先验项在总下界中的贡献。
            - `"vb"`: 包含各个时间步的变分下界术语的张量。
            - `"xstart_mse"`: 包含各个时间步的 x_0 均方误差的张量。
            - `"mse"`: 包含各个时间步的噪声均方误差的张量。

        这个函数的主要目的是计算给定时间步的模型中的变分下界，并提供了一些与变分下界相关的额外信息，
        例如先验项、每个时间步的变分下界术语和误差度量。通常，这用于评估模型的性能和训练过程。
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    这是一个辅助函数 _extract_into_tensor，用于从一个一维 NumPy 数组 arr 中提取值，根据给定的索引 timesteps，
    并将提取的值广播到指定形状 broadcast_shape 中。这通常在深度学习或数值计算中用于处理数据的不同形状和尺寸。

    函数的主要步骤如下：
        首先，将输入的一维 NumPy 数组 arr 转换为一个 PyTorch 张量（Tensor），并确保它与索引 timesteps 的设备（device）相匹配。
    这是为了将数据从 NumPy 数组转换为 PyTorch 张量，并确保它在正确的计算设备上。
        使用索引 timesteps 从 PyTorch 张量中提取特定的数值。timesteps 应该是一个表示在 arr 中要提取的元素的索引的张量。
        然后，通过在结果张量上执行广播操作，将提取的值广播到指定的形状 broadcast_shape 中。
    广播是一种在不同形状的张量之间进行元素级操作的机制，以使它们具有相同的形状，以便进行元素级运算。
        最终，函数返回一个 PyTorch 张量，其形状为 [batch_size, 1, ...]，其中 batch_size 等于 timesteps 的长度，
    而剩余的维度由 broadcast_shape 决定。
    这个函数的目的是将一维数据根据索引提取，并根据需要广播到指定的形状，以便在更复杂的计算中使用。
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
