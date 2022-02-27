from models.layers import ConvLayer, FCLayer, PoolLayer, get_layer_by_name
import tensorflow as tf


class LayerCascade:
	def __init__(self, _id, layers):
		self._id = _id
		self.layers = layers
		
		self.output_op = None
	
	@property
	def id(self):
		return self._id
	
	@id.setter
	def id(self, value):
		self._id = value
	
	@property
	def out_features_dim(self):
		for layer in self.layers[::-1]:
			if isinstance(layer, ConvLayer):
				return layer.filter_num
			elif isinstance(layer, FCLayer):
				return layer.units
		return None
	
	@property
	def depth(self):
		return sum(isinstance(layer, (ConvLayer, FCLayer)) for layer in self.layers)
	
	def get_str(self):
		layers_str = [layer.layer_str for layer in self.layers]
		return '-'.join(layers_str)
	
	def build(self, _input, densenet, store_output_op=False):
		output = _input
		with tf.variable_scope(self._id):
			for layer in self.layers:
				output = layer.build(output, densenet, store_output_op=store_output_op)
		if store_output_op:
			self.output_op = output
		return output
	
	def get_config(self):
		return {
			'_id': self._id,
			'layers': [layer.get_config() for layer in self.layers]
		}
	
	def renew_init(self, densenet):
		return {
			'_id': self._id,
			'layers': [layer.renew_init(densenet) for layer in self.layers]
		}
	
	def copy(self):
		return self.set_from_config(self.get_config(), init=self.renew_init(None))
	
	@staticmethod
	def set_from_config(config_json, init=None, return_class=True):
		_id = config_json['_id']
		layers = []
		for _i, layer_config in enumerate(config_json['layers']):
			layer_init = init['layers'][_i] if init is not None else None
			layer = get_layer_by_name(layer_config['name'])
			layers.append(layer.set_from_config(layer_config, layer_init))
		return LayerCascade(_id, layers) if return_class else (_id, layers)
	
	"""
	Network Transformation Operations
	"""
	
	def prev_widen(self, indices, magnifier, noise=None):
		for layer in self.layers:
			if isinstance(layer, (ConvLayer, FCLayer)):
				layer.prev_widen(indices, magnifier, noise=noise)
				break
			else:
				layer.prev_widen(indices, magnifier, noise=noise)
	
	def widen(self, idx, new_width, widen_type='output_dim', noise=None):
		assert idx < len(self.layers), 'Index out of range: %d' % idx
		if widen_type != 'output_dim':
			raise ValueError('%s is not supported' % widen_type)
		assert isinstance(self.layers[idx],
		                  (ConvLayer, FCLayer)), 'Operation not available'
		to_widen_layer = self.layers[idx]

		if isinstance(to_widen_layer, ConvLayer):
			indices, magnifier = to_widen_layer.widen_filters(new_filter_num=new_width, noise=noise)
		else:
			indices, magnifier = to_widen_layer.widen_units(new_units_num=new_width, noise=noise)
		after_widen_layer = None
		for _i in range(idx + 1, len(self.layers)):
			if isinstance(self.layers[_i], (ConvLayer, FCLayer)):
				self.layers[_i].prev_widen(indices, magnifier, noise=noise)
				after_widen_layer = self.layers[_i]
				break
			else:
				self.layers[_i].prev_widen(indices, magnifier, noise=noise)
		return after_widen_layer is None, indices, magnifier
	
	def deepen(self, idx, new_layer_config, input_dim):
		assert idx < len(self.layers), 'Index out of range: %d' % idx
		if new_layer_config['name'] == 'fc':
			assert idx == len(self.layers) - 1 or isinstance(self.layers[idx + 1], FCLayer), 'Invalid'
			assert isinstance(self.layers[idx], (FCLayer, PoolLayer)), 'Invalid'
			# prepare the new fc layer
			units = input_dim
			for _i in range(idx, -1, -1):
				if isinstance(self.layers[_i], FCLayer):
					units = self.layers[_i].units
					break
				elif isinstance(self.layers[_i], ConvLayer):
					units = self.layers[_i].filter_num
					break
			fc_idx = sum(isinstance(self.layers[_i], FCLayer) for _i in range(idx + 1))
			_id = 'fc_%d' % fc_idx
			# change the id of following fc layers
			for _i in range(idx + 1, len(self.layers)):
				if isinstance(self.layers[_i], FCLayer):
					self.layers[_i].id = 'fc_%d' % (fc_idx + 1)
					fc_idx += 1
			prev_layer = next(
			    (self.layers[_i] for _i in range(idx, -1, -1) if self.layers[_i].ready),
			    None,
			)
			assert prev_layer is not None, 'Invalid'
			new_fc_layer = FCLayer(_id, units, ready=False, **new_layer_config)
			# insert the new layer into the cascade
			self.layers = self.layers[:idx + 1] + [new_fc_layer] + self.layers[idx + 1:]
			return new_fc_layer, prev_layer
		elif new_layer_config['name'] == 'conv':
			assert idx == len(self.layers) - 1 or not isinstance(self.layers[idx + 1], FCLayer), 'Invalid'
			assert isinstance(self.layers[idx], (ConvLayer, FCLayer)), 'Invalid'
			# prepare the new conv layer
			filter_num = input_dim
			for _i in range(idx, -1, -1):
				if isinstance(self.layers[_i], ConvLayer):
					filter_num = self.layers[_i].filter_num
					break
			conv_idx = sum(isinstance(self.layers[_i], ConvLayer) for _i in range(idx + 1))
			_id = 'conv_%d' % conv_idx
			# change the id of following conv layers
			for _i in range(idx + 1, len(self.layers)):
				if isinstance(self.layers[_i], ConvLayer):
					self.layers[_i].id = 'conv_%d' % (conv_idx + 1)
					conv_idx += 1
			prev_layer = next(
			    (self.layers[_i] for _i in range(idx, -1, -1) if self.layers[_i].ready),
			    None,
			)
			assert prev_layer is not None, 'Invalid'
			new_conv_layer = ConvLayer(_id, filter_num, ready=False, **new_layer_config)
			self.layers = self.layers[:idx + 1] + [new_conv_layer] + self.layers[idx + 1:]
			return new_conv_layer, prev_layer
		else:
			raise ValueError('Not support to insert a %s layer' % new_layer_config['name'])
