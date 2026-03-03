import dextrous.dst_data as dst
import dextrous.metrics as metrics
import dextrous.preprocessing as dpp

if __name__ == '__main__':
    eval_with_predictions = dst.Data(
        data_path='data/mwoz2.4/valid',
        prediction_path='ex/LlamaTracker/hRadiantAurra_preds/0/predictions.csv'
    )
    jga = metrics.joint_goal_accuracy(
        eval_with_predictions, speakers={'user'}, domains={'taxi'}
    )
    print(f'Joint Goal Accuracy: {jga:.3f}')