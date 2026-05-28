import os
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Custom_BinaryTrend, Dataset_Custom_Pretrain, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'custom_binary_trend': Dataset_Custom_BinaryTrend,
    'custom_pretrain': Dataset_Custom_Pretrain,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        # Do not drop the last (possibly smaller) batch for test.
        # Otherwise test_result.csv may miss the tail days even when test_end is set.
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
        scaler_path = getattr(args, 'scaler_path', None)
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    # 对于 pretrain dataset，不使用 flag 参数
    if args.data == 'custom_pretrain':
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            size=[args.seq_len],
            features=args.features,
            timeenc=timeenc,
            freq=freq
        )
    elif flag == 'pred':
        pred_start = getattr(args, 'pred_start', None)
        pred_end = getattr(args, 'pred_end', None)
        cols = getattr(args, 'cols', None)
        if cols is not None and isinstance(cols, str):
            cols = [c.strip() for c in cols.replace(',', ' ').split() if c.strip()]
        data_set = Dataset_Pred(
            root_path=args.root_path,
            data_path=args.data_path,
            size=[args.seq_len],
            features=args.features,
            target=args.target,
            scale=True,
            inverse=False,
            timeenc=timeenc,
            freq=freq,
            cols=cols,
            scaler_path=scaler_path,
            pred_start=pred_start,
            pred_end=pred_end
        )
    else:
        # 对于 custom_binary_trend，需要传递 scaler_path 和 separate
        if args.data == 'custom_binary_trend':
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len],
                features=args.features,
                target=args.target,
                scaler_path=getattr(args, 'scaler_path', None),
                timeenc=timeenc,
                freq=freq,
                train_end=getattr(args, 'train_end', None),
                val_end=getattr(args, 'val_end', None),
                test_end=getattr(args, 'test_end', None),
                separate=getattr(args, 'separate', False)
            )
        else:
            cols = getattr(args, 'cols', None)
            if cols is not None and isinstance(cols, str):
                cols = [c.strip() for c in cols.replace(',', ' ').split() if c.strip()]
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                train_end=getattr(args, 'train_end', None),
                train_start=getattr(args, 'train_start', None),
                test_end=getattr(args, 'test_end', None),
                val_ratio=getattr(args, 'val_ratio', 0.1),
                cols=cols
            )
    
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
