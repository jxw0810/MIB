from collections import OrderedDict
import warnings
import itertools
from contextlib import contextmanager

try:
    import tensorflow.compat.v1 as tf
except ModuleNotFoundError:
    import tensorflow as tf

import tensorflow_probability as tfp

import numpy as np
import keras
from IBA.utils import WelfordEstimator, _to_saliency_map, get_tqdm
import keras.backend as K
from IBA._keras_graph import contains_activation

# expose for importing
from IBA._keras_graph import pre_softmax_tensors


class TFWelfordEstimator(WelfordEstimator):
    """
    Estimates the mean and standard derivation.
    For the algorithm see `wikipedia <https://en.wikipedia.org/wiki/
    Algorithms_for_calculating_variance#/Welford's_online_algorithm>`_.

    Args:
        feature_name (str): name of the feature tensor
        graph (tf.Graph): graph which holds the feature tensor. If ``None``,
            uses the default graph.
    """
    def __init__(self, feature_name, graph=None):
        self._feature_name = feature_name
        self._graph = graph or tf.get_default_graph()
        super().__init__()

    def fit(self, feed_dict, session=None, run_kwargs={}):
        """
        Estimates the mean and std given the inputs in ``feed_dict``.

        .. warning ::

            Ensure that your model is in eval mode. If you use keras, call
            ``K.set_learning_phase(0)``.

        Args:
            feed_dict (dict): tensorflow feed dict with model inputs.
            session (tf.Session): session to execute the model. If ``None``,
                uses the default session.
            run_kwargs (dict): additional kwargs to ``session.run``.
        """
        session = session or tf.get_default_session() or K.get_session()
        feature = self._graph.get_tensor_by_name(self._feature_name)
        feat_values = session.run(feature, feed_dict=feed_dict, **run_kwargs)
        super().fit(feat_values)

    def fit_generator(self, generator, session=None, progbar=True, run_kwargs={}):
        """
        Estimates the mean and std from the ``feed_dict`` generator.

        .. warning ::

            Ensure that your model is in eval mode. If you use keras, call
            ``K.set_learning_phase(0)``.

        Args:
            generator: yield tensorflow ``feed_dict``s.
            session (tf.Session): session to execute the model. If ``None``,
                uses the default session.
            run_kwargs (dict): additional kwargs to ``session.run``.
            progbar (bool): flag to show progress bar.
        """
        try:
            tqdm = get_tqdm()
            progbar = tqdm(generator, progbar=progbar)
        except ImportError:
            progbar = generator

        for feed_dict in progbar:
            self.fit(feed_dict, session, run_kwargs)

    def state_dict(self) -> dict:
        """Returns the estimator internal state. Can be loaded with :meth:`load_state_dict`.

        Example: ::

            state = estimator.state_dict()
            with open('estimator_state.pickle', 'wb') as f:
                pickle.dump(state, f)

            # load it

            estimator = TFWelfordEstimator(feature_name=None)
            with open('estimator_state.pickle', 'rb') as f:
                state = pickle.load(f)
                estimator.load_state_dict(state)

        """
        state = super().state_dict()
        state['feature_name'] = self._feature_name

    def load_state_dict(self, state: dict):
        """Loads estimator internal state."""
        super().load_state_dict(state)
        self._feature_mean = state['feature_mean']


def to_saliency_map(capacity, shape=None, data_format=None):
    """
    Converts the layer capacity (in nats) to a saliency map (in bits) of the given shape .

    Args:
        capacity (np.ndarray): Capacity in nats.
        shape (tuple): (height, width) of the image.
        data_format (str): ``"channels_first"`` or ``"channels_last"``. If None,
            the ``K.image_data_format()`` of keras is used.
    """
    data_format = data_format or K.image_data_format()
    return _to_saliency_map(capacity, shape, data_format)


def _kl_div(r, lambda_, mean_r, std_r):
    r_norm = (r - mean_r) / std_r

    # variance of Z, 计算Z的方差
    var_z = (1 - lambda_) ** 2

    log_var_z = tf.log(var_z)
    mu_z = r_norm * lambda_

    # Return the feature-wise KL-divergence of p(z|x) and q(z)
    capacity = -0.5 * (1 + log_var_z - mu_z**2 - var_z)
    return capacity


def _gaussian_kernel(size, std):
    """Makes 2D gaussian Kernel for convolution."""
    d = tfp.distributions.Normal(0., std)
    vals = d.prob(tf.cast(tf.range(start=-size, limit=size + 1), tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def _gaussian_blur(x, std=1.):

    # Cover 2.5 stds in both directions
    kernel_size = tf.cast((tf.round(4 * std)) * 2 + 1, tf.int32)

    kernel = _gaussian_kernel(kernel_size // 2, std)
    kernel = kernel[:, :, None, None]
    kernel = tf.tile(kernel, (1, 1, x.shape[-1], 1))

    kh = kernel_size//2

    if len(x.shape) == 4:
        x = tf.pad(x, [[0, 0], [kh, kh], [kh, kh], [0, 0]], "REFLECT")
        x_blur = tf.nn.depthwise_conv2d(
            x,
            kernel,
            strides=[1, 1, 1, 1],
            padding='VALID',
            name='blurring',
        )
    elif len(x.shape) == 3:
        x = tf.pad(x, [[0, 0], [kh, kh], [0, 0]], "REFLECT")
        kernel = kernel[:, kh+1:kh+2]
        x_extra_dim = x[:, :, None]
        x_blur = tf.nn.depthwise_conv2d(
            x_extra_dim,
            kernel,
            strides=[1, 1, 1, 1],
            padding='VALID',
            name='blurring',
        )
        x_blur = x_blur[:, :, 0]
    else:
        raise ValueError("shape not supported! got: {}".format(x.shape))
    return tf.cond(tf.math.equal(std, 0.), lambda: x, lambda: x_blur)


def model_wo_softmax(model: keras.Model):
    """Creates a new model w/o the final softmax activation.
       ``model`` must be a keras model.
    """
    return keras.models.Model(inputs=model.inputs,
                              outputs=pre_softmax_tensors(model.outputs),
                              name=model.name)


class IBALayer(keras.layers.Layer):
    """
    A keras layer that can be included in your model.
    This class should work with any model and does not copy the tensorflow graph.
    Although it is a keras layer, it should be possible to use it from other libaries.
    If you cannot alter your model definition, you have to copy the graph (use
    :class:`.IBACopy` or :class:`.IBACopyInnvestigate`).

    Example:  ::

        model = keras.Sequential()

        # add some layer
        model.add(Conv2D(64, 3, 3))
        model.add(BatchNorm())
        model.add(Activation('relu'))
        # ... more layers

        # add iba in between
        iba = IBALayer()
        model.add(iba)

        # ... more layers
        model.add(Conv2D(64, 3, 3))
        model.add(Flatten())
        model.add(Dense(10))

        # set classification cross-entropy loss
        target = iba.set_classification_loss(model.output)

        # estimate the feature mean and std.
        for imgs, _ in data_generator():
            iba.fit({model.input: imgs})

        # explain target for image
        ex_image, ex_target = get_explained_image()
        saliency_map = iba.analyze({model.input: ex_image, target: ex_target})


    Hyperparamters Paramters:
        The informational bottleneck attribution has a few hyperparameters. They
        most important parameter is the ``beta`` which controls the trade-off
        between model loss. Generally, ``beta = 10`` works well.  Other
        hyperparameters are the number of optimization ``steps``. The
        ``learning_rate`` of the optimizer.  The smoothing of the feature map
        and the minimum feature standard derviation.  All hyperparamters set in
        the constructure can be overriden in the :meth:`analyze` method or in
        the :meth:`set_default` method.

    Args:
        estimator (TFWelfordEstimator): already fitted estimator.
        feature_mean: estimated feature mean.
            Do not provide ``feature_mean_std`` and ``estimator``.
        feature_std: estimated feature std.
        feature_active: estimated active neurons. If ``feature_active[i] = 0``, the i-th  neuron
            will be set to zero and no information is added for this neuron.
        batch_size (int): Default number of samples to average the gradient.
        steps (int): Default number of iterations to optimize.
        beta (int): Default value for trade-off between model loss and information loss.
        learning_rate (float): Default learning rate of the Adam optimizer.
        min_std (float): Default minimum feature standard derivation.
        smooth_std (float): Default smoothing of the lambda parameter. Set to ``0`` to disable.
        normalize_beta (bool): Default flag to devide beta by the nubmer of feature
            neurons (default: ``True``).
        **kwargs: keras layer kwargs, see ``keras.layers.Layer``
    """
    def __init__(self, estimator=None,
                 feature_mean=None,
                 feature_std=None,
                 feature_active=None,
                 # default hyper parameters
                 batch_size=10,
                 steps=10,
                 beta=1,
                 learning_rate=1,
                 min_std=0.01,
                 smooth_std=1.,
                 normalize_beta=True,
                 **kwargs):
        self._estimator = estimator
        self._model_loss_set = False

        self._feature_mean = feature_mean
        self._feature_std = feature_std
        self._feature_active = feature_active
        self._feature_mean_std_given = (self._feature_mean is not None and
                                        self._feature_std is not None)

        self._default_hyperparams = {
            "batch_size": batch_size,
            "steps": steps,
            "beta": beta,
            "learning_rate": learning_rate,
            "min_std": min_std,
            "smooth_std": smooth_std,
            "normalize_beta": normalize_beta,
        }

        self._collect_names = []

        self._report_tensors = OrderedDict()
        self._report_tensors_first = OrderedDict()

        super().__init__(**kwargs)

    def _get_session(self, session=None):
        """ Returns session if not None or the keras or tensoflow default session.  """
        return session or keras.backend.get_session() or tf.get_default_session()

    def set_default(self, batch_size=None, steps=None, beta=None, learning_rate=None,
                    min_std=None, smooth_std=None, normalize_beta=None):
        """Updates the default hyperparamter values. """
        if batch_size is not None:
            self._default_hyperparams['batch_size'] = batch_size
        if steps is not None:
            self._default_hyperparams['steps'] = steps
        if beta is not None:
            self._default_hyperparams['beta'] = beta
        if learning_rate is not None:
            self._default_hyperparams['learning_rate'] = learning_rate
        if min_std is not None:
            self._default_hyperparams['min_std'] = min_std
        if smooth_std is not None:
            self._default_hyperparams['smooth_std'] = smooth_std
        if normalize_beta is not None:
            self._default_hyperparams['normalize_beta'] = normalize_beta

    def get_default(self):
        """Returns the default hyperparamter values."""
        return self._default_hyperparams

    # Reporting

    def _report(self, name, tensor):
        assert name not in self._report_tensors
        self._report_tensors[name] = tensor

    def _report_first(self, name, tensor):
        assert name not in self._report_tensors_first
        self._report_tensors_first[name] = tensor

    def _get_report_tensors(self):
        ret = OrderedDict()
        for name in self._collect_names:
            if name in self._report_tensors:
                ret[name] = self._report_tensors[name]
        return ret

    def _get_report_tensors_first(self):
        ret = self._get_report_tensors()
        for name in self._collect_names:
            if name in self._report_tensors_first:
                ret[name] = self._report_tensors_first[name]
        return ret

    def collect(self, *var_names):
        """
        Mark ``*var_names`` to be collected for the report.
        See :meth:`available_report_variables` for all variable names.
        """
        for name in var_names:
            assert name in self._report_tensors or name in self._report_tensors_first, \
                "not tensor found with name {}! Try one of these: {}".format(
                    name, self.available_report_variables())
        self._collect_names = var_names

    def collect_all(self):
        """
        Mark all variables to be collected for the report. If all variables are collected,
        the optimization can slow down.
        """
        self.collect(*self.available_report_variables())

    def available_report_variables(self):
        """Returns all variables that can be collected for :meth:`get_report`."""
        return sorted(list(self._report_tensors.keys()) + list(self._report_tensors_first.keys()))

    def get_report(self):
        """Returns the report for the last run."""
        return self._log

    def build(self, input_shape):
        """ Builds the keras layer given the input shape.  """
        shape = self._feature_shape = [1, ] + [int(d) for d in input_shape[1:]]

        # optimization placeholders
        self.learning_rate = tf.get_variable('learning_rate',  initializer=1.0)
        self._beta = tf.get_variable('beta', dtype=tf.float32,
                                     initializer=np.array(10.).astype(np.float32))

        self._batch_size = tf.get_variable('batch_size', dtype=tf.int32, initializer=12)

        # trained parameters
        alpha_init = 5
        self._alpha = tf.get_variable(name='alpha', initializer=alpha_init*tf.ones(shape))

        # feature map
        self._feature = tf.get_variable('feature', shape, trainable=False)

        # mean of feature map r
        self._mean_r = tf.get_variable(
            name='mean_r', trainable=False,  dtype=tf.float32, initializer=tf.zeros(shape))
        # std of feature map r
        self._std_r = tf.get_variable(
            name='std_r', trainable=False,  dtype=tf.float32, initializer=tf.zeros(shape))
        self._active_neurons = tf.get_variable(
            name='active_neurons', trainable=False,  dtype=tf.float32, initializer=tf.zeros(shape))
        # mask that indicate that no noise should be applied to a specific neuron
        self._pass_mask = tf.get_variable(name='pass_mask', trainable=False,
                                          dtype=tf.float32, initializer=tf.zeros(shape))

        # flag to restrict the flow
        self._restrict_flow = tf.get_variable(name='restrict_flow', trainable=False,
                                              dtype=tf.bool, initializer=False)

        self._use_layer_input = tf.get_variable(name='use_layer_input', trainable=False,
                                                dtype=tf.bool, initializer=False)

        # min standard derivation per neuron
        self._min_std_r = tf.get_variable('min_std_r', dtype=tf.float32, initializer=0.1)
        # kernel size for gaussian blur
        self._smooth_std = tf.get_variable('smooth_std', dtype=tf.float32, initializer=1.)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """ Returns the ``input_shape``. """
        return input_shape

    def call(self, inputs) -> tf.Tensor:
        """ Returns the output tensor. You can enable the restriction of
        the information flow with: :meth:`.restrict_flow`. """
        if self._estimator is None and not self._feature_mean_std_given:
            self._estimator = TFWelfordEstimator(inputs.name)

        feature = tf.cond(self._use_layer_input, lambda: inputs, lambda: self._feature)

        tile_batch_size = tf.cond(self._use_layer_input,
                                  lambda: 1,
                                  lambda: self._batch_size)
        tile_shape = [tile_batch_size, ] + [1] * (len(self._feature_shape) - 1)
        # tf.tile对张量(Tensor)进行扩展的
        R = tf.tile(feature, tile_shape)
        pass_mask = tf.tile(self._pass_mask, tile_shape)
        restrict_mask = 1 - pass_mask

        std_r_min = tf.maximum(self._std_r, self._min_std_r)

        # std_too_low = tf.cast(self.std_x < self.min_std_r, tf.float32)
        # std_large_enough = tf.cast(self.std_x >= self.min_std_r, tf.float32)

        lambda_pre_blur = tf.sigmoid(self._alpha)
        # λ = _gaussian_blur(lambda_pre_blur, std=self._smooth_std)
        λ = lambda_pre_blur

        # std normal noise N(0, 1)
        noise_shape = [tile_batch_size] + R.get_shape().as_list()[1:]
        std_normal = tf.random.normal(noise_shape)

        # ε ~ N(μ_r, σ_r)
        ε = std_r_min * std_normal + self._mean_r

        Z = λ * R + (1 - λ) * ε

        # let all information through for neurons in pass_mask
        Z_with_passing = restrict_mask * Z + pass_mask * R
        Z_with_passing *= self._active_neurons

        output = tf.cond(self._restrict_flow, lambda: Z_with_passing, lambda: inputs)
        # save capacityies

        self._capacity = (_kl_div(R, λ, self._mean_r, std_r_min) *
                          restrict_mask * self._active_neurons)
        # 如果是
        self._capacity_no_nans = tf.where(tf.is_nan(self._capacity),
                                          tf.zeros_like(self._capacity),
                                          self._capacity)
        self._capacity_mean = tf.reduce_sum(self._capacity_no_nans) / tf.reduce_sum(restrict_mask)

        # save tensors for report
        self._report('lambda_pre_blur', lambda_pre_blur)
        self._report('lambda',  λ)
        self._report('eps', ε)
        self._report('alpha', self._alpha)
        self._report('capacity', self._capacity)
        self._report('capacity_no_nans', self._capacity_no_nans)
        self._report('capacity_mean', self._capacity_mean)

        self._report('perturbed_feature', Z)
        self._report('perturbed_feature_passing', Z_with_passing)

        self._report_first('feature', self._feature)
        self._report_first('feature_mean', self._mean_r)
        self._report_first('feature_std', self._std_r)
        self._report_first('pass_mask', pass_mask)

        return output

    @contextmanager
    def restrict_flow(self, session=None):
        """
        Context manager to restrict the flow of the layer.  Useful to estimate
        model output when noise is added.  If the flow restirction is enabled,
        you can only call the model with a single sample (batch size = 1).

        Example: ::

            capacity = iba.analyze({model.input: x})
            # computes logits using all information
            logits = model.predict(x)
            with iba.restrict_flow():
                # computes logits using only a subset of all information
                logits_restricted = model.predict(x)
        """
        session = self._get_session(session)
        old_value = session.run(self._restrict_flow)
        session.run(tf.assign(self._restrict_flow, True))
        yield
        session.run(tf.assign(self._restrict_flow, old_value))

    # Set model loss

    def set_classification_loss(self, logits, optimizer_cls=tf.train.AdamOptimizer) -> tf.Tensor:
        """
        Creates a cross-entropy loss from the logit tensors.
        Returns the target tensor.

        Example: ::

            iba.set_classification_loss(model.output)

        You have to ensure that the final layer of ``model`` does not applies a softmax.
        For keras models, you can remove a softmax activation using :func:`model_wo_softmax`.
        """
        self.target = tf.get_variable('iba_target', dtype=tf.int32, initializer=[1])

        target_one_hot = tf.one_hot(self.target, depth=logits.shape[-1])
        loss_ce = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=target_one_hot,
            logits=logits,
            name='cross_entropy'
        )
        loss_ce_mean = tf.reduce_mean(loss_ce)
        self._report('logits', logits)
        self._report('cross_entropy', loss_ce)
        self.set_model_loss(loss_ce_mean, optimizer_cls)
        return self.target

    def set_model_loss(self, model_loss, optimizer_cls=tf.train.AdamOptimizer):
        """
        Sets the model loss for the final objective ``model_loss + beta * capacity_mean``.
        When build the ``model_loss``, ensure you are using the copied graph.

        Example: ::

            with iba.copied_session_and_graph_as_default():
                iba.get_copied_outputs()

        """
        self._optimizer = optimizer_cls(learning_rate=self._default_hyperparams['learning_rate'])
        information_loss = self._capacity_mean
        loss = model_loss + self._beta * information_loss
        self._optimizer_step = self._optimizer.minimize(loss, var_list=[self._alpha])

        self._report('loss', loss)
        self._report('model_loss', model_loss)
        self._report('information_loss', information_loss)
        self._report('grad_loss_wrt_alpha', tf.gradients(loss, self._alpha)[0])
        self._model_loss_set = True

    # Fit std and mean estimator

    def fit(self, feed_dict, session=None, run_kwargs={}):
        """
        Estimate the feature mean and std from the given feed_dict.

        .. warning ::

            Ensure that your model is in eval mode. If you use keras, call
            ``K.set_learning_phase(0)``.

        Args:
            generator: Yields feed_dict with all inputs
            n_samples: Stop after ``n_samples``
            session: use this session. If ``None`` use default session.
            run_kwargs: additional kwargs to ``session.run``.

        Example: ::

            # input is a tensorflow placeholder  of your model
            input = tf.placeholder(tf.float32, name='input')

            X, y = load_data_batch()
            iba.fit({input: X})

        Where ``input`` is a tensorflow placeholder and ``X`` an input numpy array.
        """

        self._estimator.fit(feed_dict, session, run_kwargs)

    def fit_generator(self, generator, n_samples=5000, progbar=True, session=None, run_kwargs={}):
        """
        Estimates the feature mean and std from the generator.


        .. warning ::

            Ensure that your model is in eval mode. If you use keras, call
            ``K.set_learning_phase(0)``.

        Args:
            generator: Yields ``feed_dict`` s with inputs to all placeholders.
            n_samples (int): Stop after ``n_samples``.
            session (tf.Session): tf session to use. If ``None`` use default session.
            run_kwargs (dict): additional kwargs to ``session.run``.
        """

        try:
            tqdm = get_tqdm()
            gen = tqdm(generator, disable=not progbar, desc="[Fit Estimator]")
        except ImportError:
            if progbar:
                warnings.warn("Cannot load tqdm! Sorry, no progress bar")
            gen = generator

        for step, feed_dict in enumerate(gen):
            self._estimator.fit(feed_dict, session=session, run_kwargs=run_kwargs)
            if self._estimator.n_samples() >= n_samples:
                break

    def analyze(self, feed_dict,
                batch_size=None,
                steps=None,
                beta=None,
                learning_rate=None,
                min_std=None,
                smooth_std=None,
                normalize_beta=None,
                session=None,
                pass_mask=None,
                progbar=False) -> np.ndarray:
        """
        Returns the transmitted information per feature. See :func:`to_saliency_map` to convert the
        intermediate capacites to a visual saliency map.

        Args:
            feed_dict (dict): TensorFlow feed_dict providing your model inputs.
            batch_size (int): number of samples to average the gradient.
            steps (int): number of iterations to optimize.
            beta (int): trade-off parameter between model loss and information loss.
            learning_rate (float): Learning rate of the Adam optimizer.
            min_std (float): Minimum feature standard derivation.
            smooth_std (float): Smoothing of the lambda. Set to ``0`` to disable.
            normalize_beta (bool): Devide beta by the nubmer of feature neurons
                (default: ``True``).
            session (tf.Session): TensorFlow session to run the optimization.
            pass_mask (np.array): same shape as the feature map.
                ``pass_mask`` masks neurons which are always passed to the next layer.
                No noise is added if ``pass_mask == 0``.  For example, it might
                be usefull if a variable lenght sequence is zero-padded.
            progbar (bool): Flag to display progressbar.
        """
        session = self._get_session(session)
        feature = session.run(self.input, feed_dict=feed_dict)

        return self._analyze_feature(
            feature,
            feed_dict,
            pass_mask=pass_mask,
            batch_size=batch_size,
            steps=steps,
            beta=beta,
            learning_rate=learning_rate,
            min_std=min_std,
            smooth_std=smooth_std,
            normalize_beta=normalize_beta,
            session=session,
            progbar=False)

    def _analyze_feature(self,
                         feature,
                         feed_dict,
                         batch_size=None,
                         steps=None,
                         beta=None,
                         learning_rate=0.1,
                         min_std=None,
                         smooth_std=None,
                         normalize_beta=True,
                         pass_mask=None,
                         session=None,
                         progbar=False):
        if session is None:
            session = keras.backend.get_session()

        batch_size = batch_size or self._default_hyperparams['batch_size']
        steps = steps or self._default_hyperparams['steps']
        beta = beta or self._default_hyperparams['beta']
        learning_rate = learning_rate or self._default_hyperparams['learning_rate']
        min_std = min_std or self._default_hyperparams['min_std']
        smooth_std = smooth_std or self._default_hyperparams['smooth_std']
        normalize_beta = normalize_beta or self._default_hyperparams['normalize_beta']

        if not hasattr(self, '_optimizer'):
            raise ValueError("Optimizer not build yet! You have to specify your model loss "
                             "by calling the set_model_loss method.")
        self._log = OrderedDict()

        if not normalize_beta:
            # we use the mean of the capacity, which is equivalent to dividing by k =h *w*c.
            # therefore, we have to denormalize beta: β = β*h*w*c.
            beta = beta * np.prod(feature.shape)

        if self._feature_mean_std_given:
            feature_mean = self._feature_mean
            feature_std = self._feature_std
            feature_active = self._feature_active
        else:
            assert self._estimator.n_samples() > 0
            feature_mean = self._estimator.mean()
            feature_std = self._estimator.std()
            feature_active = self._estimator.active_neurons()

        def maybe_unsqueeze(x):
            if len(self._mean_r.shape) == len(x.shape) + 1:
                return x[None]
            else:
                return x

        feature_mean = maybe_unsqueeze(feature_mean)
        feature_std = maybe_unsqueeze(feature_std)
        feature_active = maybe_unsqueeze(feature_active)

        # set hyperparameters
        assigns = [
            self._alpha.initializer,
            tf.variables_initializer(self._optimizer.variables()),
            tf.assign(self._mean_r, feature_mean, name='assign_feature_mean'),
            tf.assign(self._std_r, feature_std, name='assign_feature_std'),
            tf.assign(self._active_neurons, feature_active, name='assign_feature_std'),
            tf.assign(self._feature, feature, name='assign_feature'),
            tf.assign(self._beta, beta, name='assign_beta'),
            tf.assign(self._smooth_std, smooth_std, name='assign_smooth_std'),
            tf.assign(self._min_std_r, min_std, name='assign_min_std'),
            tf.assign(self.learning_rate, learning_rate, name='assign_lr'),
            tf.assign(self._restrict_flow, True, name='assign_restrict_flow'),
            tf.assign(self._use_layer_input, False, name='assign_use_layer_input'),
        ]
        if pass_mask is None:
            pass_mask = tf.zeros_like(self._pass_mask)
        assigns.append(tf.assign(self._pass_mask, pass_mask))
        session.run(assigns)

        report_tensors = self._get_report_tensors()
        report_tensors_first = self._get_report_tensors_first()

        if len(report_tensors_first) > 0:
            outs = session.run(list(report_tensors_first.values()),
                               feed_dict=feed_dict)
            self._log['init'] = OrderedDict(zip(report_tensors_first.keys(), outs))
        else:
            self._log['init'] = OrderedDict()

        try:
            tqdm = get_tqdm()
            steps_progbar = tqdm(range(steps), total=steps,
                                 disable=not progbar, desc="[Fit Estimator]")
        except ImportError:
            if progbar:
                warnings.warn("Cannot load tqdm! Sorry, no progress bar")
            steps_progbar = range(steps)

        for step in steps_progbar:
            outs = session.run(
                [self._optimizer_step] + list(report_tensors.values()),
                feed_dict=feed_dict)
            self._log[step] = OrderedDict(zip(report_tensors_first.keys(), outs[1:]))

        final_report_tensors = list(report_tensors.values())
        final_report_tensor_names = list(report_tensors.keys())

        if 'capacity' not in report_tensors:
            final_report_tensors.append(self._capacity)
            final_report_tensor_names.append('capacity')


        vals = session.run(final_report_tensors, feed_dict=feed_dict)
        self._log['final'] = OrderedDict(zip(final_report_tensor_names, vals))

        # reset flags up
        session.run([
            tf.assign(self._restrict_flow, False, name='assign_restrict_flow'),
            tf.assign(self._use_layer_input, True, name='assign_use_layer_input'),
        ])
        return self._log['final']['capacity'][0]

    def state_dict(self):
        """
        Returns the current layer state.
        """
        return {
            'estimator': self._estimator.state_dict(),
            'feature_mean': self._feature_mean,
            'feature_std': self._feature_std,
            'feature_active': self._feature_active,
            'default_hyperparams': self._default_hyperparams
        }

    def load_state_dict(self, state):
        """
        Load the given ``state``.
        """
        self._estimator.load_state_dict(state['estimator'])
        self._feature_mean = state['feature_mean']
        self._feature_std = state['feature_std']
        self._default_hyperparams = state['default_hyperparams']
