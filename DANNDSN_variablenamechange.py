import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cPickle as pkl
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import urllib
import os
import tarfile
import skimage
import skimage.io
import skimage.transform

# PARAMETERS
batch_size = 100
training_epochs = 50
every_epoch = 1

def compose_image(digit, background):
    """Difference-blend a digit and a random patch from a background image."""
    w, h, _ = background.shape
    dw, dh, _ = digit.shape
    x = np.random.randint(0, w - dw)
    y = np.random.randint(0, h - dh)
    bg = background[x:x+dw, y:y+dh]
    return np.abs(bg - digit).astype(np.uint8)

def mnist_to_img(x):
    """Binarize MNIST digit and convert to RGB."""
    x = (x > 0).astype(np.float32)
    d = x.reshape([28, 28, 1]) * 255
    return np.concatenate([d, d, d], 2)

def create_mnistm(X):
    """
    Give an array of MNIST digits, blend random background patches to
    build the MNIST-M dataset as described in
    http://jmlr.org/papers/volume17/15-239/15-239.pdf
    """
    X_ = np.zeros([X.shape[0], 28, 28, 3], np.uint8)
    for i in range(X.shape[0]):
        bg_img = rand.choice(background_data)
        d = mnist_to_img(X[i])
        d = compose_image(d, bg_img)
        X_[i] = d
    return X_
print ("FUNCTIONS READY")

mnistm_name = 'data/mnistm_data2.pkl'
if os.path.isfile(mnistm_name):
    print ("[%s] ALREADY EXISTS. " % (mnistm_name))
else:
    mnist = input_data.read_data_sets('data')
    # OPEN BSDS500
    f = tarfile.open(filename)
    train_files = []
    for name in f.getnames():
        if name.startswith('BSR/BSDS500/data/images/train/'):
            train_files.append(name)
    print ("WE HAVE [%d] TRAIN FILES" % (len(train_files)))
    # GET BACKGROUND
    print ("GET BACKGROUND FOR MNIST-M")
    background_data = []
    for name in train_files:
        try:
            fp = f.extractfile(name)
            bg_img = skimage.io.imread(fp)
            background_data.append(bg_img)
        except:
            continue
    print ("WE HAVE [%d] BACKGROUND DATA" % (len(background_data)))
    rand = np.random.RandomState(42)
    print ("BUILDING TRAIN SET...")
    train = create_mnistm(mnist.train.images)
    print ("BUILDING TEST SET...")
    test = create_mnistm(mnist.test.images)
    print ("BUILDING VALIDATION SET...")
    valid = create_mnistm(mnist.validation.images)
    # SAVE
    print ("SAVE MNISTM DATA TO %s" % (mnistm_name))
    with open(mnistm_name, 'w') as f:
        pkl.dump({ 'train': train, 'test': test, 'valid': valid }, f, -1)
    print ("DONE")

print ("LOADING MNIST")
mnist        = input_data.read_data_sets('data', one_hot=True)
mnist_train  = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
mnist_train  = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
mnist_test   = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
mnist_test   = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
mnist_train_label = mnist.train.labels
mnist_test_label = mnist.test.labels


print ("LOADING MNIST-M")
mnistm_name  = 'data/mnistm_data2.pkl'
mnistm       = pkl.load(open(mnistm_name))
mnistm_train = mnistm['train']
mnistm_test  = mnistm['test']
mnistm_valid = mnistm['valid']
mnistm_train_label = mnist_train_label
mnistm_test_label = mnist_test_label
print ("GENERATING DOMAIN DATA")

total_train        = np.vstack([mnist_train, mnistm_train])
total_test         = np.vstack([mnist_test, mnistm_test])
ntrain             = mnist_train.shape[0]
ntest              = mnist_test.shape[0]
total_train_domain = np.vstack([np.tile([1., 0.], [ntrain, 1]), np.tile([0., 1.], [ntrain, 1])])
total_test_domain  = np.vstack([np.tile([1., 0.], [ntest, 1]), np.tile([0., 1.], [ntest, 1])])
n_total_train      = total_train.shape[0]
n_total_test       = total_test.shape[0]

# GET PIXEL MEAN
pixel_mean = np.vstack([mnist_train, mnistm_train]).mean((0, 1, 2))

# PLOT IMAGES
def imshow_grid(images, shape=[2, 8]):
    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)
    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])
    plt.show()
imshow_grid(mnist_train, shape=[5, 10])
imshow_grid(mnistm_train, shape=[5, 10])


def print_npshape(x, name):
    print("SHAPE OF %s IS %s" % (name, x.shape,))


# SOURCE AND TARGET DATA
#source_train_img = mnist_train
#source_train_label = mnist_train_label
source_train_img = np.concatenate((mnist_train,mnist_train),axis=0)
source_train_label = np.concatenate((mnist_train_label,mnist_train_label),axis=0)

source_test_img = mnist_test
source_test_label = mnist_test_label

#target_train_img = mnistm_train
#target_train_label= mnistm_train_label
target_train_img = np.concatenate((mnistm_train,mnistm_train),axis=0)
target_train_label= np.concatenate((mnistm_train_label,mnistm_train_label),axis=0)

target_test_img = mnistm_test
target_test_label = mnistm_test_label




# DOMAIN ADVERSARIAL TRAINING
domain_train_img = total_train
domain_train_label = total_train_domain

imgshape = source_train_img.shape[1:4]
labelshape = source_train_label.shape[1]

print_npshape(source_train_img, "source_train_img")
print_npshape(source_train_label, "source_train_label")
print_npshape(source_test_img, "source_test_img")
print_npshape(source_test_label, "source_test_label")
print_npshape(target_test_img, "target_test_img")
print_npshape(target_test_label, "target_test_label")
print_npshape(domain_train_img, "domain_train_img")
print_npshape(domain_train_label, "domain_train_label")

print
imgshape
print
labelshape

class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0
    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls
        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            #return [tf.neg(grad) * l]
            return [tf.negative(grad) * l]
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)
        self.num_calls += 1
        return y

flip_gradient = FlipGradientBuilder()

##Place Holder
source  = tf.placeholder(tf.uint8, [None, imgshape[0], imgshape[1], imgshape[2]])
target  = tf.placeholder(tf.uint8, [None, imgshape[0], imgshape[1], imgshape[2]])
source_target  = tf.placeholder(tf.uint8, [None, imgshape[0], imgshape[1], imgshape[2]])

y  = tf.placeholder(tf.float32, [None, labelshape])
d  = tf.placeholder(tf.float32, [None, 2]) # DOMAIN LABEL
lr = tf.placeholder(tf.float32, [])
dw = tf.placeholder(tf.float32, [])

# FEATURE EXTRACTOR
def shared_encoder(x, name='feat_ext', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        x = (tf.cast(x, tf.float32) - pixel_mean) / 255.
        net = slim.conv2d(x, 32, [5, 5], scope = 'conv1_shared_encoder')
        net = slim.max_pool2d(net, [2, 2], scope='pool1_shared_encoder')
        net = slim.conv2d(net, 48, [5, 5], scope='conv2_shared_encoder')
        net = slim.max_pool2d(net, [2, 2], scope='pool2_shared_encoder')
        feat = slim.flatten(net, scope='flat_shared_encoder')
    return feat

#Private Target Encoder
def private_target_encoder(x, name='priviate_target_encoder', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        x = (tf.cast(x, tf.float32) - pixel_mean) / 255.
        net = slim.conv2d(x, 32, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net, 48, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        feat = slim.flatten(net, scope='flat')
    return feat

#Private Source Encoder
def private_source_encoder(x, name='priviate_source_encoder', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        x = (tf.cast(x, tf.float32) - pixel_mean) / 255.
        net = slim.conv2d(x, 32, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net, 48, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        feat = slim.flatten(net, scope='flat')
    return feat


# CLASS PREDICTION
def class_pred_net(feat, name='class_pred', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        net = slim.fully_connected(feat, 100, scope='fc1')
        net = slim.fully_connected(net, 100, scope='fc2')
        net = slim.fully_connected(net, 10, activation_fn = None, scope='out')
    return net

# DOMAIN PREDICTION
def domain_pred_net(feat, name='domain_pred', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        feat = flip_gradient(feat, dw) # GRADIENT REVERSAL
        net = slim.fully_connected(feat, 100, scope='fc1')
        net = slim.fully_connected(net, 2, activation_fn = None, scope='out')
    return net

def shared_decoder(feat,height,width,channels,reuse=False, name='shared_decoder'):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        net = slim.fully_connected(feat, 600, scope='fc1_shared_decoder')
        net = tf.reshape(net, [-1, 10, 10, 6])
        net = slim.conv2d(net, 32, [5, 5], scope='conv1_1_shared_decoder')
        net = tf.image.resize_nearest_neighbor(net, (16, 16))
        net = slim.conv2d(net, 32, [5, 5], scope='conv2_1_shared_decoder')
        net = tf.image.resize_nearest_neighbor(net, (height, width))
        net = slim.conv2d(net, channels, [5, 5], scope='conv3_2_shared_decoder')
    return net


# DIFFERENCE LOSS
def difference_loss(private_samples, shared_samples, weight=1.0, name=''):
  private_samples -= tf.reduce_mean(private_samples, 0)
  shared_samples -= tf.reduce_mean(shared_samples, 0)
  private_samples = tf.nn.l2_normalize(private_samples, 1)
  shared_samples = tf.nn.l2_normalize(shared_samples, 1)
  correlation_matrix = tf.matmul( private_samples, shared_samples, transpose_a=True)
  cost = tf.reduce_mean(tf.square(correlation_matrix)) * weight
  cost = tf.where(cost > 0, cost, 0, name='value')
  #tf.summary.scalar('losses/Difference Loss {}'.format(name),cost)
  assert_op = tf.Assert(tf.is_finite(cost), [cost])
  with tf.control_dependencies([assert_op]):
     tf.losses.add_loss(cost)
  return cost

# DOMAIN ADVERSARIAL NEURAL NETWORK
feat_ext_dann    = shared_encoder(source_target, name='dann_feat_ext')
class_pred_dann  = class_pred_net(feat_ext_dann, name='dann_class_pred')
domain_pred_dann = domain_pred_net(feat_ext_dann, name='dann_domain_pred')

# NAIVE CONVOLUTIONAL NEURAL NETWORK
feat_ext_cnn     = shared_encoder(source, name='cnn_feat_ext')
#feat_ext_cnn     = shared_encoder(mnist_train, name='cnn_feat_ext', reuse=True)
class_pred_cnn   = class_pred_net(feat_ext_cnn, name='cnn_class_pred')

#Private & Shared Encoder
source_private_feat = private_source_encoder(source, name='source_private')
target_private_feat = private_target_encoder(target, name='target_private')
source_shared_feat = shared_encoder(source, name='source_shared')
target_shared_feat = shared_encoder(target, name='target_shared')


#Input for Decoder
target_concat_feat = tf.concat([target_shared_feat,target_private_feat],1)
source_concat_feat = tf.concat([source_shared_feat,source_private_feat],1)

#Decoder
#target_recon = small_decoder(target_concat_feat,28,28,3)
target_recon = shared_decoder(target_concat_feat,28,28,3)
source_recon = shared_decoder(source_concat_feat,28,28,3,reuse=True)

############################################################################################################################################################

print ("MODEL READY")

t_weights = tf.trainable_variables()

# TOTAL WEIGHTS
# print ("   TOTAL WEIGHT LIST")
# for i in range(len(t_weights)):  print ("[%2d/%2d] [%s]" % (i, len(t_weights), t_weights[i]))
"""
# FEATURE EXTRACTOR + CLASS PREDICTOR
print ("   WEIGHT LIST FOR CLASS PREDICTOR")
w_class = []
for i in range(len(t_weights)):
    if t_weights[i].name[:9] == 'dann_feat' or t_weights[i].name[:10] == 'dann_class':
        w_class.append(tf.nn.l2_loss(t_weights[i]))
        print ("[%s]    \t ADDED TO W_CLASS LIST" % (t_weights[i].name))

l2loss_dann_class = tf.add_n(w_class)

# FEATURE EXTRACTOR + DOMAIN CLASSIFIER
print ("\n   WEIGHT LIST FOR DOMAIN CLASSIFIER")
w_domain = []

for i in range(len(t_weights)):
    if t_weights[i].name[:8] == 'dann_feat' or t_weights[i].name[:11] == 'dann_domain':
        w_domain.append(tf.nn.l2_loss(t_weights[i]))
        print ("[%s]    \t ADDED TO W_DOMAIN LIST" % (t_weights[i].name))


l2loss_cnn_domain = tf.add_n(w_domain)
"""


# FUNCTIONS FOR DANN
class_loss_dann  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=class_pred_dann, labels=y))
domain_loss_dann = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=domain_pred_dann, labels=d))
target_recon_loss = tf.reduce_mean(tf.pow((tf.cast(target,tf.float32) - target_recon),2))
source_recon_loss = tf.reduce_mean(tf.pow((tf.cast(source,tf.float32)  - source_recon),2))
target_diff_loss = difference_loss(target_private_feat,target_shared_feat)
source_diff_loss = difference_loss(source_private_feat,source_shared_feat)

############################################################################################################################################################
losses = class_loss_dann + target_recon_loss + source_recon_loss + target_diff_loss + source_diff_loss
optm_class_dann  = tf.train.MomentumOptimizer(lr, 0.9).minimize(losses)

optm_domain_dann = tf.train.MomentumOptimizer(lr, 0.9).minimize(domain_loss_dann)


accr_class_dann  = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(class_pred_dann, 1), tf.arg_max(y, 1)), tf.float32))
accr_domain_dann = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(domain_pred_dann, 1), tf.arg_max(d, 1)), tf.float32))

# FUNCTIONS FOR CNN
class_loss_cnn   = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=class_pred_cnn, labels=y))
optm_class_cnn   = tf.train.MomentumOptimizer(lr, 0.9).minimize(class_loss_cnn)
accr_class_cnn   = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(class_pred_cnn, 1), tf.arg_max(y, 1)), tf.float32))
print ("FUNCTIONS READY")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
tf.set_random_seed(0)
init = tf.global_variables_initializer()
sess.run(init)
print ("SESSION OPENED")



# PARAMETERS
batch_size = 100
training_epochs = 50
every_epoch = 1
#num_batch = int(ntrain / batch_size) + 1
num_batch = int(ntrain/ batch_size)
total_iter = training_epochs * num_batch
for epoch in range(training_epochs):
    randpermlist = np.random.permutation(ntrain)
    for i in range(num_batch):
        # REVERSAL WEIGHT AND LEARNING RATE SCHEDULE
        curriter = epoch * num_batch + i
        p = float(curriter) / float(total_iter)
        dw_val = 2. / (1. + np.exp(-10. * p)) - 1
        lr_val = 0.01 / (1. + 10 * p) ** 0.75
        # OPTIMIZE DANN: CLASS-CLASSIFIER
        #randidx_class = randpermlist[i * batch_size:min((i + 1) * batch_size, ntrain - 1)]
        randidx_class = randpermlist[i * batch_size:min((i + 1) * batch_size, ntrain-1)]
        randidx_domain = randpermlist[i * batch_size:min((i + 1) * batch_size, ntrain-1)]
        # randidx_domain = np.random.permutation(n_total_train)[:batch_size]
        batch_source = source_train_img[randidx_class]
        batch_target = target_train_img[randidx_class]
        batch_x_domain = total_train[randidx_domain]
        batch_y_class = source_train_label[randidx_class, :]
        #print(batch_source.shape)
        #print(batch_target.shape)
        #print(batch_x_domain.shape)
        #print(batch_y_class.shape)
        feeds_class = {source: batch_source,source_target: batch_x_domain, target:batch_target, y: batch_y_class, lr: lr_val, dw: dw_val}
        #feeds_class = {source: batch_source, target:batch_target, y: batch_y_class, lr: lr_val, dw: dw_val}
        _, lossclass_val_dann = sess.run([optm_class_dann, class_loss_dann], feed_dict=feeds_class)
        # OPTIMIZE DANN: DOMAIN-CLASSIFER
        randidx_domain = np.random.permutation(n_total_train)[:batch_size]
        batch_x_domain = total_train[randidx_domain]
        batch_d_domain = total_train_domain[randidx_domain, :]
        feeds_domain = {source_target: batch_x_domain, d: batch_d_domain, lr: lr_val, dw: dw_val}
        _, lossdomain_val_dann = sess.run([optm_domain_dann, domain_loss_dann], feed_dict=feeds_domain)
        # OPTIMIZE DANN: CLASS-CLASSIFIER
        _, lossclass_val_cnn = sess.run([optm_class_cnn, class_loss_cnn], feed_dict=feeds_class)
    if epoch % every_epoch == 0:
        # CHECK BOTH LOSSES
        print("[%d/%d][%d/%d] p: %.3f lossclass_val: %.3e, lossdomain_val: %.3e"
              % (epoch, training_epochs, curriter, total_iter, p, lossdomain_val_dann, lossclass_val_dann))
        # CHECK ACCUARACIES OF BOTH SOURCE AND TARGET
        #feed_source = {source: batch_source, source_target: batch_x_domain, target:batch_target, y: batch_y_class, lr: lr_val, dw: dw_val}
        #feed_target = {source: batch_source, source_target: batch_x_domain, target: batch_target, y: batch_y_class,lr: lr_val, dw: dw_val}
        feed_source = {source_target: source_test_img, y: source_test_label}
        feed_target = {source_target: target_test_img, y: target_test_label}
        feed_source_cnn = {source: source_test_img, y: source_test_label}
        feed_target_cnn = {source: target_test_img, y: target_test_label}
        accr_source_dann = sess.run(accr_class_dann, feed_dict=feed_source)
        accr_target_dann = sess.run(accr_class_dann, feed_dict=feed_target)
        accr_source_cnn = sess.run(accr_class_cnn, feed_dict=feed_source_cnn)
        accr_target_cnn = sess.run(accr_class_cnn, feed_dict=feed_target_cnn)
        print(" DANN: SOURCE ACCURACY: %.3f TARGET ACCURACY: %.3f"
              % (accr_source_dann, accr_target_dann))
        print(" CNN: SOURCE ACCURACY: %.3f TARGET ACCURACY: %.3f"
              % (accr_source_cnn, accr_target_cnn))
