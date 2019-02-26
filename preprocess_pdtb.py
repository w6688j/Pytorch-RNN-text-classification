from util import create_pdtb_tsv_file

if __name__ == '__main__':
    create_pdtb_tsv_file('pdtb/train.txt', 'data/train_pdtb.tsv')
    create_pdtb_tsv_file('pdtb/test.txt', 'data/test_pdtb.tsv')
