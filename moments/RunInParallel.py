"""
For evaluation of gradients by parallel finite-difference calculation.

This code has not been tested thoroughly.
"""
import logging
logging.basicConfig()
logger = logging.getLogger('RunInParallel')

import cPickle, os, sys, traceback

# With some versions of pypar and on some systems, importing pypar changes
# the current working directory. We save it here so we can change back.
currdir = os.getcwd()
try:
    import pypar
    HAVE_PYPAR = True
    num_procs = pypar.size()
    my_rank = pypar.rank()
    my_host = pypar.get_processor_name()
    import atexit
    atexit.register(pypar.finalize)
except ImportError:
    HAVE_PYPAR = False
    num_procs = 1
    my_rank = 0
    import socket
    my_host = socket.gethostname()
os.chdir(currdir)
logger.debug('Node %i is on host %s.' % (my_rank, my_host))

class Statement:
    """
    Class for sending Python statements to workers.
    """
    def __init__(self, statement, locals={}):
        self.statement = statement
        self.locals = locals

def eval_func_over_list(func_str, x_todo, *args):
    return [eval('%s(x, *args)' % func_str) for x in x_todo]

while my_rank != 0:
    # Wait for a message
    message = pypar.receive(source=0)

    # If the message is a SystemExit exception, exit the code.
    if isinstance(message, SystemExit):
        sys.exit()

    # Exception handling:
    #    If we catch any exception, it's probably a bug in the code. Print
    #      a nice traceback, save results, and exit the code.
    try:
        if isinstance(message, Statement):
            command, msg_locals = message.statement, message.locals
            locals().update(msg_locals)
            exec(command)
        else:
            command, msg_locals = message 
            locals().update(msg_locals)
            result = eval(command)
            pypar.send(result, 0)
    except:
        # Assemble and print a nice traceback
        tb = traceback.format_exception(sys.exc_type, sys.exc_value, 
                                        sys.exc_traceback)
        logger.critical(os.getcwd())
        logger.critical(('node %i:'%my_rank).join(tb))
        save_to = 'node_%i_crash.bp' % my_rank
        logger.critical("node %i: Command being run was: %s."
                        % (my_rank, command))
        dump_file = file(save_to, 'w')
        cPickle.dump(msg_locals, dump_file)
        dump_file.close()
        logger.critical("node %i: Corresponding locals saved to %s."
                        % (my_rank, save_to))
        sys.exit()

def stop_workers():
    """
    Send all workers the command to exit the program.
    """
    for worker in range(1, num_procs):
        pypar.send(SystemExit(), worker)

if my_rank == 0:
    import atexit
    atexit.register(stop_workers)

def statement_to_all_workers(statement, locals={}):
    """
    Send a Python statement to all workers for execution.
    """
    for worker in range(1, num_procs):
        pypar.send(Statement(statement, locals), worker)

# Ensure that all workers are working in the correct directory.
if my_rank == 0:
    statement_to_all_workers("os.chdir('%s')" % currdir)
    statement_to_all_workers("sys.path = master_path", {'master_path':sys.path})

import numpy
def make_parallel_gradient_func(func_str, epsilon=1e-8, send_grad_args=True):
    def grad_func(x0, *grad_args):
        sys.stdout.flush()
        x0 = numpy.asarray(x0)

        # Generate a list of parameter sets to evaluate the function at
        x_todo = [x0]
        for ii in range(len(x0)):
            eps = numpy.zeros(len(x0), numpy.float_)
            eps[ii] = epsilon
            x_todo.append(x0 + eps)

        # Break that list apart into work for each node
        x_by_node = []
        for node_ii in range(num_procs):
            x_by_node.append(x_todo[node_ii::num_procs])

        # Master sends orders to all other nodes.
        command = 'eval_func_over_list(func_str, x_todo, *grad_args)'
        for node_ii in range(1, num_procs):
            x_this_node = x_by_node[node_ii]
            arguments = {'func_str': func_str,
                         'x_todo': x_this_node}
            if send_grad_args:
                arguments['grad_args'] = grad_args
            pypar.send((command, arguments), node_ii)
        sys.stdout.flush()

        # This will hold the function evaluations done by each node.
        vals_by_node = []
        # The master node does its share of the work now
        vals_by_node.append(eval_func_over_list(func_str, x_by_node[0], 
                                                *grad_args))
        # Now receive the work done by each of the other nodes
        for node_ii in range(1, num_procs):
            vals_by_node.append(pypar.receive(node_ii))

        # Reform the function value list that's broken apart by node.
        func_evals = numpy.zeros(len(x0)+1, numpy.float_)
        for node_ii,vals_this_node in enumerate(vals_by_node):
            func_evals[node_ii::num_procs] = vals_this_node

        # Now calculate the gradient
        grad = numpy.zeros(len(x0), numpy.float_)
        f0 = func_evals[0]
        for ii,func_val in enumerate(func_evals[1:]):
            grad[ii] = (func_val - f0)/epsilon

        return grad
    return grad_func
