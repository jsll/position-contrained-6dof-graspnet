from options.test_options import TestOptions
from models import create_model


def run_test(epoch=-1, name="", writer=None, dataset_test=None):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    opt.name = name
    #dataset = DataLoader(opt)
    model = create_model(opt)
    # test
    point_clouds = []
    for data in dataset_test:
        model.set_input(data)
        ncorrect, nexamples = model.test()
        point_clouds.append(model.get_random_grasp_and_point_cloud())
        writer.update_counter(ncorrect, nexamples)
    writer.calculate_accuracy()
    writer.print_acc(epoch)
    writer.plot_acc(epoch)
    writer.plot_grasps(point_clouds, epoch)


if __name__ == '__main__':
    run_test()
