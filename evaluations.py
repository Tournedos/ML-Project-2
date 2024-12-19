import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def single_curve(surv_array):
    stat = np.full(surv_array.shape, True)
    time, survival_prob, conf_int = kaplan_meier_estimator(
        stat, surv_array, conf_type="log-log")
    plt.step(time, survival_prob, where="post")
    plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
    plt.ylim(0, 1)
    plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")


def quant_evaluation(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rsq = r2_score(y_true, y_pred)
    print(f'Model Performances: ')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R-squared: {rsq}')

def double_curve(y_true, y_pred):
    larr = np.array(y_pred)
    stat = np.full(larr.shape, True)
    time, survival_prob, conf_int = kaplan_meier_estimator(
        stat, larr, conf_type="log-log"
    )

    larrtr = np.array(y_true)
    stattr = np.full(larrtr.shape, True)
    time2, survival_prob2, conf_int2 = kaplan_meier_estimator(
        stattr, larrtr, conf_type="log-log"
    )

    plt.step(time, survival_prob, where="post", label='model prediction')
    plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
    plt.step(time2, survival_prob2, where="post", label='real times')
    plt.fill_between(time2, conf_int2[0], conf_int2[1], alpha=0.25, step="post")
    plt.legend()
    plt.ylim(0, 1)
    plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
