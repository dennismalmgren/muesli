import sys
from torchrl.envs import (
    DoubleToFloat,
    RenameTransform,
    ExcludeTransform,
    SqueezeTransform,
    Compose,
    CatFrames
)
from torchrl.data.datasets.d4rl import D4RLExperienceReplay
from torchrl.data import (
    RoundRobinWriter,
    SliceSampler,
    TensorStorage,
)

def get_project_root_path_vscode():
    trace_object = getattr(sys, 'gettrace', lambda: None)() #vscode debugging
    if trace_object is not None:
        return "../../../"
    else:
        return "../../../../"
    
def load_d4rl_rb(batch_size:int):
    d2f = DoubleToFloat()
    # rename = RenameTransform(
    #     in_keys=[
    #         "action",
    #         "observation",
    #         ("next", "observation"),
    #     ],
    #     out_keys=[
    #         "action_cat",
    #         "observation_cat",
    #         ("next", "observation_cat"),
    #     ],
    # )
    squeeze = SqueezeTransform(
        squeeze_dim=-1,
        in_keys=[
            "done",
            "terminated",
            "truncated",
            ("next", "done"),
            ("next", "terminated"),
            ("next", "truncated"),
        ]        
    )    
    exclude = ExcludeTransform(
        "terminal",
        "info",
        ("next", "timeout"),
        ("next", "terminal"),
        ("next", "observation"),
        ("next", "info"),
    )
    cf = CatFrames(2, in_keys=["observation"], dim=-1, padding="constant")
    transforms = Compose(
        d2f,
        squeeze,
       # rename,
        exclude,
        cf
    )
    data = D4RLExperienceReplay(
        dataset_id="walker2d-medium-v0",
        split_trajs=False,
        batch_size=batch_size,
        sampler=SliceSampler(slice_len=2, traj_key="traj_ids"),  # SamplerWithoutReplacement(drop_last=False),
        transform=None,
        use_truncated_as_done=True,
        direct_download=True,
        prefetch=4,
        writer=RoundRobinWriter(),
    )
    data_memmap = data[:]
    with data_memmap.unlock_():
        data_memmap = data_memmap.reshape(-1)
        data._storage = TensorStorage(data_memmap)
        print('ok')
    for t in transforms:
        data.append_transform(t)
    return data
