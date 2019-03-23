import tensorflow as tf

dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
#sess = tf.Session()
sess = tf.InteractiveSession()
tf.global_variables_initializer()


nest = iterator.get_next()
for i in range(100):
	value = sess.run(nest)
	print(value)
