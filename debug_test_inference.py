import zarr

from inference import measure_stats, predict_aff, full_inference, thresholding
from inference2 import Thresholding


def test_local_prediction():
    input_path = "/cajal/nvmescratch/projects/NISB/base/val/seed100/data.zarr"
    img_data = zarr.open(input_path, mode="r")["img"]

    model_path = "/cajal/scratch/projects/misc/zuzur/ss3/debug1GPU-seed0-batch_size1-small_size128/default/checkpoints/epoch=0-step=70000.ckpt"
    from BANIS import BANIS

    model = BANIS.load_from_checkpoint(model_path)

    all_stats = {}

    for chunk_cube_size in [200, 400, 512, 750, 1024, 1500, 3000]:
        measured_predict_aff = measure_stats(predict_aff)

        result, stats = measured_predict_aff(img_data, model, chunk_cube_size=chunk_cube_size, compute_backend="local",
                        zarr_path=f"/cajal/scratch/projects/misc/zuzur/test{chunk_cube_size}.zarr", do_overlap=True,
                        prediction_channels=3, divide=255, small_size=model.hparams.small_size)

        all_stats[chunk_cube_size] = stats
        print(f"chunk size {chunk_cube_size}: {stats}")

    print(all_stats)
    for (value, stat) in all_stats.items():
        print(f"{value}: {stat}")


def test_slurm_prediction():
    input_path = "/cajal/nvmescratch/projects/NISB/base/val/seed100/data.zarr"
    img_data = zarr.open(input_path, mode="r")["img"]

    model_path = "/cajal/scratch/projects/misc/zuzur/ss3/debug1GPU-seed0-batch_size1-small_size128/default/checkpoints/epoch=0-step=70000.ckpt"
    from BANIS import BANIS
    model = BANIS.load_from_checkpoint(model_path)

    measured_predict_aff = measure_stats(predict_aff)
    # only one run - runtime dependent on number of available slurm nodes
    result, stats = measured_predict_aff(img_data, model_path=model_path, chunk_cube_size=512, compute_backend="slurm",
                    zarr_path=f"/cajal/scratch/projects/misc/zuzur/test_slurm.zarr", do_overlap=True,
                    prediction_channels=3, divide=255, small_size=model.hparams.small_size)

    print(stats)

def test_full_inference():
    input_path = "/cajal/nvmescratch/projects/NISB/base/val/seed100/data.zarr"
    img_data = zarr.open(input_path, mode="r")["img"]

    model_path = "/cajal/scratch/projects/misc/zuzur/ss3/debug1GPU-seed0-batch_size1-small_size128/default/checkpoints/epoch=0-step=70000.ckpt"
    from BANIS import BANIS
    model = BANIS.load_from_checkpoint(model_path)

    full_inference(img_data, model_path, thr=0.7685)

def test_thresholding():
    aff = zarr.open("/cajal/scratch/projects/misc/zuzur/skeleton_recall/rerun_base/dsbase_s0_a0_25-03-20_19-44-08-067760/pred_aff_val_6.zarr/")
    postprocessor = Thresholding(1024, "local", 0.7685)
    postprocessor.aff_to_seg(aff, zarr_path="test2.zarr")
    #thresholding(aff, 0.7685, "test2.zarr", 1024, "local")

if __name__ == "__main__":
    test_thresholding()
