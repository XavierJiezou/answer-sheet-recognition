import cv2
import numpy as np
from skimage import measure


def baweraopen(image, size):
    """
    @image:单通道二值图，数据类型uint8
    @size:欲去除区域大小(黑底上的白区域)
    """
    output = image.copy()
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    for i in range(1, nlabels - 1):
        regions_size = stats[i, 4]
        if regions_size < size:
            x0 = stats[i, 0]
            y0 = stats[i, 1]
            x1 = stats[i, 0] + stats[i, 2]
            y1 = stats[i, 1] + stats[i, 3]
            for row in range(y0, y1):
                for col in range(x0, x1):
                    if labels[row, col] == i:
                        output[row, col] = 0
    return output


def answerGet(path, part):
    r = cv2.imread(path, 0)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    res = cv2.morphologyEx(r, cv2.MORPH_BLACKHAT, se, iterations=1)
    m, n = res.shape
    ret, rbw = cv2.threshold(res, 20, 255, cv2.THRESH_BINARY)
    # cv2.imshow("bw", rbw)
    lines = cv2.HoughLinesP(rbw, 1, np.pi / 180, 40,
                            minLineLength=200, maxLineGap=50)
    an = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x1 = float(x1)
        x2 = float(x2)
        y1 = float(y1)
        y2 = float(y2)
        if x2 - x1 == 0:
            result = 90
        elif y2 - y1 == 0:
            result = 0
        else:
            # 计算斜率
            k = -(y2 - y1) / (x2 - x1)
            # 求反正切，再将得到的弧度转换为度
            result = np.arctan(k) * 57.29577
        an.append(result)
    sum_ang = 0
    for i in an:
        if i > 90:
            i -= 180
        elif i == 90:
            continue
        sum_ang += i
        # print(i)
    anp = float(sum_ang / len(an))
    # print(anp)
    M = cv2.getRotationMatrix2D(((n - 1) / 2.0, (m - 1) / 2.0), -anp, 1)
    res = cv2.warpAffine(rbw, M, (n, m))
    # cv2.imshow("spin", res)
    # res = cv2.medianBlur(res, 3)
    lr = 0
    lr2 = 0
    lu = 0
    rl = 0
    rd = 0
    for i in range(m):
        flag = 0
        count = 0
        for j in range(n):
            if res[i, j] == 255:
                count += 1
            if count >= 20:
                lu = i

                flag = 1
                break
        if flag == 1:
            break

    for i in range(m - 1, 0, -1):
        flag = 0
        count = 0
        for j in range(n):
            if res[i, j] == 255:
                count += 1
            if count >= 20:
                rd = i
                flag = 1
                break
        if flag == 1:
            break

    for j in range(n):
        flag = 0
        count = 0
        for i in range(m):
            if res[i, j] == 255:
                count += 1
            if count >= 20:
                lr = j
                flag = 1
                break
        if flag == 1:
            break

    for j in range(n - 1, 0, -1):
        flag = 0
        count = 0
        for i in range(m):
            if res[i, j] == 255:
                count += 1
            if count >= 20:
                rl = j
                flag = 1
                break
        if flag == 1:
            break
    lr2 = lu
    sinL = int((rl - lr) / 11)
    sinH = int((rd - lu) / part)

    temp = []
    count = 1
    # res2 = np.zeros([m, n])
    # for l_ in range(m):
    #     for h_ in range(n):
    #         res2[l_, h_]=res[l_, h_]
    res2 = res.copy()
    # cv2.imshow("res2", res2)
    for ind in range(0, part):
        for jn in range(0, 11):
            lel = lr + jn * sinL
            leh = lr2 + ind * sinH
            flag = 0

            for k2 in range(leh + 1, lr2 + (ind + 1) * sinH + 1):

                cmt = 1
                for k in range(lel + int(sinH * 0.05), lr + (jn + 1) * sinL - int(sinH * 0.05) + 1):
                    if res[k2, k] == 255 and res[k2, k - 1] == 255:
                        cmt += 1

                if 2 <= cmt <= 10:

                    for k in range(lel, lr + (jn + 1) * sinL):
                        res[k2, k] = 0

    res = baweraopen(res, 700)
    # cv2.imshow("lt", res)
    ro6 = np.zeros([m, n])

    for l_ in range(m):
        for h_ in range(n):
            if res2[l_, h_] == 255 and res[l_, h_] == 255:
                ro6[l_, h_] = 0
            if res2[l_, h_] == 255 and res[l_, h_] == 0:
                ro6[l_, h_] = 255
            if res2[l_, h_] == 0 and res[l_, h_] == 0:
                ro6[l_, h_] = 0
            if res2[l_, h_] == 0 and res[l_, h_] == 255:
                ro6[l_, h_] = 0

    for ind in range(0, part):
        for jn in range(0, 11):
            lel = lr + jn * sinL
            leh = lr2 + ind * sinH
            flag = 0
            tem = np.zeros([sinH + 1, sinL + 1])
            kt = 0
            k2t = 0
            for k2 in range(leh + 5, lr2 + (ind + 1) * sinH + 4):
                for k in range(lel, lr + (jn + 1) * sinL + 1):
                    tem[kt, k2t] = ro6[k2, k]
                    k2t += 1
                k2t = 0

                kt += 1
            temp.append(tem)

    cnt2 = 0

    Flag = 0
    kk = 0

    answer = []
    num = 0
    for te in temp:
        kk += 1

        if Flag != 0:
            if kk == 12 or kk == 34:
                continue
            # print(num)
            num += 1
            v = {}
            ma = 0
            mk = 0
            L = measure.label(te, connectivity=2, return_num=True)
            L2 = L[0].copy()
            L2 = np.mat(L2)
            m_, n_ = L2.shape
            for l_ in range(m_):
                for h_ in range(n_):
                    if L2[l_, h_] == 0:
                        continue
                    if L2[l_, h_] not in v.keys():
                        v[L2[l_, h_]] = 1
                    else:
                        v[L2[l_, h_]] = v[L2[l_, h_]] + 1
                        if v[L2[l_, h_]] > ma:
                            ma = v[L2[l_, h_]]
                            mk = L2[l_, h_]

            L = measure.label(te, connectivity=2, return_num=True)
            L2 = L[0].copy()
            m_, n_ = L2.shape
            L2_ = np.zeros([m_, n_])
            for l_ in range(m_):
                for h_ in range(n_):
                    if L2[l_, h_] == mk:
                        L2_[l_, h_] = 255
                    else:
                        L2_[l_, h_] = 0

            # cv2.imshow(str(num),L2_)
            answer.append(L2_)
        if kk % 11 == 0 and Flag == 0:
            Flag = 1
            continue
        if kk % 11 == 0 and Flag == 1:
            Flag = 0
            continue

    # cv2.imshow("ss", res)
    # cv2.waitKey(0)
    return answer # 现在answer里存的就是分割出来的选项

    
# answer = answerGet("2.png")
# for i, j in enumerate(answer):
#     print(j.shape)
#     cv2.imwrite('pics/'+str(i+1)+'.png', j)