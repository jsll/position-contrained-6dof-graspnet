import time
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model
from utils.writer import Writer
from test import run_test


def main():
    opt = TrainOptions().parse()
    if opt == None:
        return

    data_loader = DataLoader(opt)
    training_dataset, test_dataset, _ = data_loader.split_dataset(opt.dataset_split_ratio)

    dataset_train = data_loader.create_dataloader(training_dataset, shuffle_batches=not opt.serial_batches)
    dataset_test = data_loader.create_dataloader(test_dataset, shuffle_batches=False)
    dataset_train_size = len(training_dataset)
    dataset_test_size = len(test_dataset)
    print('#train images = %d' % dataset_train_size)
    print('#test images = %d' % dataset_test_size)

    dataset_size = dataset_train_size * opt.num_grasps_per_object
    model = create_model(opt)
    writer = Writer(opt)
    total_steps = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset_train):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            if total_steps % opt.print_freq == 0:
                loss_types = []
                loss = [
                        model.loss, model.kl_loss, model.reconstruction_loss,
                    ]
                loss_types = [
                        "total_loss", "kl_loss", "reconstruction_loss"
                    ]
                t = (time.time() - iter_start_time) / opt.batch_size
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data,
                                            loss_types)
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size,
                                 loss_types)
                writer.plot_grasps([model.get_random_grasp_and_point_cloud()], epoch, "train")
            if i % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('latest', epoch)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_network('latest', epoch)
            #model.save_network(str(epoch), epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay,
               time.time() - epoch_start_time))
        model.update_learning_rate()
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        if epoch % opt.run_test_freq == 0:
            run_test(epoch, name=opt.name, writer=writer, dataset_test=dataset_test)
    writer.close()


if __name__ == '__main__':
    main()
