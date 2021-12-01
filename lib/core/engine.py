class Engine(object):
    def __init__(self):
        self.hooks = {}

    def hook(self, name, state):

        if name in self.hooks:
            self.hooks[name](state)

    def adjust_learning_rate(optimizer, shrink_factor):
        print("\nDECAYING learning rate.")
        # logging.info("\nDECAYING learning rate.")
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * shrink_factor
        print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
        # logging.info("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


    def train(self, network, iterator, maxepoch, optimizer, scheduler):
        state = {
            'network': network,
            'iterator': iterator,
            'maxepoch': maxepoch,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'epoch': 0,
            't': 0,
            'train': True,
        }

        self.hook('on_start', state)
        while state['epoch'] < state['maxepoch']:
            self.hook('on_start_epoch', state)
            #if (state['epoch'] + 1) % 10 == 0:
            #    self.adjust_learning_rate(state['optimizer'],0.6)
            state['scheduler'].step()
            for sample in state['iterator']:
                state['sample'] = sample
                self.hook('on_sample', state)

                def closure():
                    loss, output = state['network'](state['sample'])
                    state['output'] = output
                    state['loss'] = loss
                    loss.backward()
                    self.hook('on_forward', state)
                    # to free memory in save_for_backward
                    state['output'] = None
                    state['loss'] = None
                    return loss

                state['optimizer'].zero_grad()
                state['optimizer'].step(closure)
                self.hook('on_update', state)
                state['t'] += 1
            state['epoch'] += 1
            self.hook('on_end_epoch', state)
        self.hook('on_end', state)
        return state

    def test(self, network, iterator, split):
        state = {
            'network': network,
            'iterator': iterator,
            'split': split,
            't': 0,
            'train': False,
        }

        self.hook('on_test_start', state)
        for sample in state['iterator']:
            state['sample'] = sample
            self.hook('on_test_sample', state)

            def closure():
                loss, output = state['network'](state['sample'])
                state['output'] = output
                state['loss'] = loss
                self.hook('on_test_forward', state)
                # to free memory in save_for_backward
                state['output'] = None
                state['loss'] = None

            closure()
            state['t'] += 1
        self.hook('on_test_end', state)
        return state
