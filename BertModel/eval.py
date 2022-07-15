
model.load_state_dict(torch.load(output_model_file))
model.to(device)
res_path='/home/user1/PangBo/GNNTrafficClassification/BertModel/result/precision_recall.csv'
res_file_temp = open(res_path, 'w', encoding='utf-8', newline="")
graph_file = csv.writer(res_file_temp)
# 损失函数准备
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

# test the model
test_loss, test_acc, test_report, test_auc, all_idx, all_labels, all_preds = evaluate_save(
    model, test_dataloader, criterion, device, label_list)
print("-------------- Test -------------")
print(f'\t  Loss: {test_loss: .3f} | Acc: {test_acc*100: .3f} % | AUC:{test_auc}')
graph_file.writerow([test_acc])
for label in label_list:
    print('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(
        label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))
    graph_file.writerow([test_report[label], test_report[label]['recall'],test_report[label]['f1-score']])
print_list = ['macro avg', 'weighted avg']

for label in print_list:
    print('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(
        label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))
