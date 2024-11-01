import numpy as np

if __name__ == '__main__':
    x_all = np.load(f"./BioSeq/dataset/tfbind/tfbind-x-all.npy")
    y_all = np.load(f"./BioSeq/dataset/tfbind/tfbind-y-all.npy")
    
# Add print statements to show the first and last 100 data points
    print("First 100 data points of x_all:", x_all[:100])  # Display first 100 data points
    print("Last 100 data points of x_all:", x_all[-100:])  # Display last 100 data points
    print("First 100 data points of y_all:", y_all[:100])  # Display first 100 data points
    print("Last 100 data points of y_all:", y_all[-100:])  # Display last 100 data points

    for wt_idx in range(6,100):
        # wt_idx = np.random.choice(len(x_all))
        wt = x_all[wt_idx]
        wt_score = y_all[wt_idx]  # 0.4312
        collected_samples = [wt]
        collected_scores = [wt_score]
    
        local_max = 0.
        for x, y in zip(collected_samples, collected_scores):
            for i, seq in enumerate(x_all):
                if 0 < ((seq - x) != 0).sum() < 3 and y_all[i] < wt_score:
                    collected_samples.append(seq)
                    collected_scores.append(y_all[i])
                elif 0 < ((seq - x) != 0).sum() < 3 and y_all[i] > local_max:
                    local_max = y_all[i]
            if len(collected_samples) > 1000:
                break
        print(f"local_max: {local_max}")

         # Add print statements to show the first and last 100 data points for collected_samples and collected_scores
        print("First 100 collected_samples:", collected_samples[:100])  # Display first 100 collected samples
        print("Last 100 collected_samples:", collected_samples[-100:])  # Display last 100 collected samples
        print("First 100 collected_scores:", collected_scores[:100])  # Display first 100 collected scores
        print("Last 100 collected_scores:", collected_scores[-100:])  # Display last 100 collected scores
        print(f"local_max: {local_max}")
        print(f"len(collected_samples): {len(collected_samples)}")
        print(f"len(collected_scores): {len(collected_scores)}")
        if local_max < 0.95: 
            break
        
    # np.save('dataset/tfbind/local-tfbind-x-init2.npy', np.array(collected_samples)) #0 .9322764194120315
    # np.save('dataset/tfbind/local-tfbind-y-init2.npy', collected_scores)