import pandas as pd

if __name__ == '__main__':
    path = "../result/test2/0.txt"
    headers = ['id', 'mips', 'ram', 'bandwidth', 'disk']
    df = pd.read_csv(path, header=None, delimiter=',')
    df.columns = headers
    print(f"df: {df}")
    # df_mean = df.mean(axis=0)
    # df_mean.to_csv("../test.txt")
    df_mean = pd.DataFrame(columns=headers)
    df_mean = df_mean.append(df.mean(axis=0), ignore_index=True)
    df_mean['id'][0] = 0
    print(f"df_mean:\n {df_mean}")
    print(f"type(df_mean): {type(df_mean)}")
