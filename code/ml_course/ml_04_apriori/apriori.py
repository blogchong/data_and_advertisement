#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy import *
import json


def print_data_info(note_list):
    """
    进行相关特征数据的打印，简单的统计分析
    :param note_list: 特征特征处理之后的共现数据组
    :param count_all: 总记录数
    :param count_1: 特征记录数
    :return: 无
    """

    count_2 = 0
    count_3 = 0
    count_4 = 0
    count_5 = 0
    count_6 = 0
    count_7 = 0
    for i in note_list:
        if (len(i)) == 2:
            count_2 += 1
        if (len(i)) == 3:
            count_3 += 1
        if (len(i)) == 4:
            count_4 += 1
        if (len(i)) == 5:
            count_5 += 1
        if (len(i)) == 6:
            count_6 += 1
        if (len(i)) == 7:
            count_7 += 1

    print('-----------------------------------------')
    print(f"特征共现数据组个数（即特征数大于等于2的数据记录数）：{len(note_list)}")
    print('-------------------')
    print(f"特征共现数据整体分布：")
    print(f"    2个特征共现数据组[{count_2}]")
    print(f"    3个特征共现数据组[{count_3}]")
    print(f"    4个特征共现数据组[{count_4}]")
    print(f"    5个特征共现数据组[{count_5}]")
    print(f"    6个特征共现数据组[{count_6}]")
    print(f"    7个特征共现数据组[{count_7}]")
    print('-----------------------------------------')


def load_data_set(file):
    """
    加载数据，并将数据处理成关联数据组，过滤无效数据（包括正常数据、包括单一特征数据）
    :param file: 文件路径
    :return: 一个处理好的特征数据组
    """
    note_list = []
    count_all = 0
    count_1 = 0

    count_A = 0
    count_B = 0
    count_C = 0
    count_D = 0
    count_E = 0
    count_F = 0
    count_G = 0

    with open(file) as lines:
        array = lines.readlines()
        for i in array:
            count_all += 1
            for j in json.loads(i):
                notes = j["labels"]
                note = []
                if len(notes) == 8:
                    if notes[0] == 1:
                        note.append("A")
                        count_A += 1
                    if notes[1] == 1:
                        note.append("B")
                        count_B += 1
                    if notes[2] == 1:
                        note.append("C")
                        count_C += 1
                    if notes[3] == 1:
                        note.append("D")
                        count_D += 1
                    if notes[4] == 1:
                        note.append("E")
                        count_E += 1
                    if notes[5] == 1:
                        note.append("F")
                        count_F += 1
                    if notes[6] == 1:
                        note.append("G")
                        count_G += 1
                    if notes[7] == 0:
                        count_1 += 1
                # 过滤掉单一数据组
                if len(note) > 1:
                    note_list.append(note)

    print('-----------------------------------------')
    print(f"总数据数：{count_all}")
    print(f"正常的数据数：{count_all - count_1}")
    print(f"含有特征的数据数：{count_1}")

    print('-----------------------------------------')
    print(f"各特征统计：")
    print(f"    A特征（A）数：[{count_A}]")
    print(f"    B特征（B）数：[{count_B}]")
    print(f"    C特征（C）数：[{count_C}]")
    print(f"    D特征（D）数：[{count_D}]")
    print(f"    E特征（E）数：[{count_E}]")
    print(f"    F特征（F）数：[{count_F}]")
    print(f"    G特征（G）数：[{count_G}]")

    print_data_info(note_list)

    return note_list


def create_c1(data_set):
    """
    构造单项C1的对应字典
    :param data_set: 数据集
    :return: c1集合，并且通过fozenset进行格式化
    """
    c1 = []
    for transaction in data_set:
        for item in transaction:
            if not [item] in c1:
                c1.append([item])
    c1.sort()
    # 映射为frozenset唯一性的，可使用其构造字典
    return list(map(frozenset, c1))


def scan_d(d_list, ck, min_support):
    """
    # 从候选K项集到频繁K项集（支持度计算），输出频繁项集和支持度
    :param d_list: 原始数据集（整体共现数据集）
    :param ck: 当前候选K项集
    :param min_support:最小支持度
    :return:频繁项集K
    """
    ss_cnt = {}
    for t_id in d_list:
        for can in ck:
            # 判断can是否t_id的子集
            if can.issubset(t_id):
                # 判断重复项，计算K项的共现频率
                if not can in ss_cnt:
                    ss_cnt[can] = 1
                else:
                    ss_cnt[can] += 1
    # 计算总项记录数D
    num_items = float(len(d_list))
    ret_list = []
    support_data = {}
    for key in ss_cnt:
        support = ss_cnt[key] / num_items
        # 满足最小支持度的，则输出，输出频繁项集+支持度
        if support >= min_support:
            ret_list.insert(0, key)
            support_data[key] = support
    return ret_list, support_data


def dict2list(dic: dict):
    """
    dict字典转换为list
    :param dic: 字典
    :return: list
    """

    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst


def cal_support(d_list, ck, min_support):
    """
    返回当前轮次K对应的频繁项集C，以及对应C的支持度
    :param d_list: 数据集
    :param ck: 当前K轮次对应的所有频繁项集C
    :param min_support: 最小支持度
    :return:
    """
    dict_sup = {}
    # 第一层遍历，遍历数据集date，拿到每个数据元组
    for i in d_list:
        # 第二层遍历，遍历CK集合，判断ck集合中的元组对是不是在原始元组对中（是否子集）
        for j in ck:
            if j.issubset(i):
                # 对CK中的元组进行计数，后续用于计算支持度
                if not j in dict_sup:
                    dict_sup[j] = 1
                else:
                    dict_sup[j] += 1
    # 公式里的D集合，用于计算支持度的分母
    sum_count = float(len(d_list))
    support_data = {}
    # 用于保存CK下频繁元组对对应的绝对个数
    re_list = []
    for i in dict_sup:
        temp_sup = dict_sup[i] / sum_count
        # 过滤不满足最低支持度的CK集合
        if temp_sup >= min_support:
            re_list.append(i)
            # 此处可设置返回全部的支持度数据（或者频繁项集的支持度数据）
            support_data[i] = temp_sup
    # 返回当前K轮次的频繁项集C，以及对应项集C的支持度
    return re_list, support_data


def apriori_gen(lk, k):
    """
    改进的剪枝算法,移除长度超过k的，以及非k-1子项的枝，创建候选K项集 ##lk为频繁K项集
    :param lk: 其中lk为对应k-1轮次,CK集合的list
    :param k: k为当前轮次超出范围的个数k
    :return:
    """
    ret_list = []
    len_lk = len(lk)
    for i in range(len_lk):
        # 嵌套遍历，i-> ((i+1)->n)
        for j in range(i + 1, len_lk):
            l1 = list(lk[i])[:k - 2]
            l2 = list(lk[j])[:k - 2]
            l1.sort()
            l2.sort()
            # 前k-1项相等，则可相乘，这样可防止重复项出现
            if l1 == l2:
                #  进行剪枝（a1为k项集中的一个元素，b为它的所有k-1项子集）
                a = lk[i] | lk[j]
                # a为frozenset()集合， a1转换为list
                a1 = list(a)
                b = []
                # 遍历取出每一个元素，转换为set，依次从a1中剔除该元素，并加入到b中
                for q in range(len(a1)):
                    t = [a1[q]]
                    tt = frozenset(set(a1) - set(t))
                    b.append(tt)
                t = 0
                for w in b:
                    # 当b（即所有k-1项子集）都是lk（频繁的）的子集，则保留，否则删除
                    if w in lk:
                        t += 1
                if t == len(b):
                    ret_list.append(b[0] | b[1])
    return ret_list


def apriori(data_set, min_support=0.2):
    """
    apriori算法调用入口
    :param data_set: 数据集
    :param min_support: 最小过滤的支持度
    :return: 频繁项集和支持数据
    """
    c1 = create_c1(data_set)
    # 使用list()转换为列表
    d_list = list(map(set, data_set))
    # 把C1项集的所有支持度进行输出
    l1, support_data = cal_support(d_list, c1, min_support)

    support_set = {}
    for k in support_data:
        key = list(k)[0]
        score = round(support_data[k], 4)
        support_set[key] = score

    # 打印输出单个项集对应的支持度
    print('---------------------------------------------')
    print(f"单特征因素对应的支持度如下：")
    print(sorted(dict2list(support_data), key=lambda x: x[1], reverse=True))
    print('---------------------------------------------')
    # 按照第0个元素降序排列

    # 加列表框，使得1项集为一个单独元素，这里的l1是C1对应的list
    l_list = [l1]
    k = 2
    ck_all = 0
    # l_list存储了所有从C1-CK的频繁项集的list，元素是一个list
    while (len(l_list[k - 2]) > 0):
        # 进行剪枝（移除长度大于k的合并项，以及k-1项非k项子集的分枝，但不做支持度计算）
        ck = apriori_gen(l_list[k - 2], k)
        ck_all += len(ck)

        # 遍历CK项集，根据支持度输出频繁项集
        lk, sup_k = scan_d(d_list, ck, min_support)

        # 通过update把K项集新增道support_data中
        support_data.update(sup_k)
        # 把K项集也更新道l_list中
        l_list.append(lk)
        k += 1
    # 删除最后一个空集
    del l_list[-1]
    # l为频繁项集，为一个列表，1，2，3项集分别为一个元素

    print('------------------------------------------')
    print(f"总项集个数为：{ck_all}")
    print('------------------------------------------')

    return l_list, support_data


def get_sub_set(from_list, to_list):
    """
    生成所有的项集的子集
    :param from_list: 原始集合
    :param to_list: 输出集合
    :return: 无
    """
    for i in range(len(from_list)):
        t = [from_list[i]]
        tt = frozenset(set(from_list) - set(t))
        if not tt in to_list:
            to_list.append(tt)
            tt = list(tt)
            if len(tt) > 1:
                # 递归调用，不断地在to_list中新增子集
                get_sub_set(tt, to_list)


def cal_conf(freq_set, sub_set, support_data, rule_list, min_conf=0.05, min_lift=0.05):
    """
    计算最小置信度等相关信息
    :param freq_set: 当前项集
    :param sub_set: 所有子集
    :param support_data: 支持度数据
    :param rule_list: 输出集
    :param min_conf: 最小支持度
    :param min_lift: 最小提升度
    :return:
    """
    for con_seq in sub_set:
        # 计算置信度
        conf = support_data[freq_set] / support_data[freq_set - con_seq]
        # 提升度lift计算lift = p(a & b) / p(a)*p(b)
        lift = support_data[freq_set] / (support_data[con_seq] * support_data[freq_set - con_seq])
        if conf >= min_conf and lift > min_lift:
            rule_list.append((freq_set, freq_set - con_seq, con_seq, support_data[freq_set], conf, lift))


def gen_rule(l_list, support_data, min_conf=0.7, min_lift=0.5):
    """
    生成规则，频繁项集，以及对应项集之间的支持度和置信度等相关信息
    :param l_list: 频繁项集
    :param support_data: 频繁项集以及对应的支持度
    :param min_conf: 最小置信度
    :param min_list: 最小
    :return:
    """
    bigrule_list = []
    # 从二项集开始计算
    for i in range(1, len(l_list)):
        # freq_set为所有的k项集
        for freq_set in l_list[i]:
            # 求该三项集的所有非空子集，1项集，2项集，直到k-1项集，用h1表示，为list类型,里面为frozenset类型，
            h1 = list(freq_set)
            all_subset = []
            # 生成所有的子集
            get_sub_set(h1, all_subset)
            # 输出项集/支持度/置信度
            cal_conf(freq_set, all_subset, support_data, bigrule_list, min_conf, min_lift)
    return bigrule_list


if __name__ == '__main__':
    file = './data/train.json'
    data_set = load_data_set(file)
    # 返回频繁项集，以及频繁项集对应的支持度
    l_list, support_data = apriori(data_set, min_support=0.1)
    rule = gen_rule(l_list, support_data, min_conf=0.2, min_lift=0.5)

    # 按支持度排序
    rule_sort = sorted(rule, key=lambda x: x[3], reverse=True)

    for (freq_set, freq_set_from, con_seq, support, conf, lift) in rule_sort:
        print('=======================================================================================================')
        print(freq_set_from, '-->',
              con_seq, ' | 支持度',
              round(support, 4),
              ' | 置信度：', round(conf, 4))

			  