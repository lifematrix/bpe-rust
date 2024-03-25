
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use dirs;
use std::time::Instant;
use std::sync::Mutex;
use std::hash::Hash;
use num_cpus;


#[derive(Debug)]
struct VocabEntry {
    units: Vec<String>,
    freq: i32,
    uset: HashSet<String>,
}


/*
fn get_vocab<P: AsRef<Path>>(path: P) -> io::Result<HashMap<String, u32>> {
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);

    let mut vocab = HashMap::new();
    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 2 {
            let word = parts[0].to_owned() + " </w>";
            let freq: u32 = parts[1].parse().expect("Frequency is not a valid u32");
            vocab.insert(word, freq);
        } else {
            eprintln!("Invalid line format: {}", line);
        }
    }
    Ok(vocab)
}
 */

fn read_vocab<P: AsRef<Path>>(vocab_fname: P, first_n: isize) -> io::Result<HashMap<String, VocabEntry>> {
    let fn_start = Instant::now();

    let file = File::open(vocab_fname)?;
    let reader = io::BufReader::new(file);

    let mut vocab = HashMap::new();

    for (idx, line) in reader.lines().enumerate() {
        if first_n !=-1 && (idx as isize) >= first_n {
            break;
        }
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 2 {
            let word = parts[0];
            let freq: i32 = parts[1].parse().expect("Frequency is not an integer");

            let mut units: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            units.push("</w>".to_string());
            let uset: HashSet<String> = units.iter().cloned().collect();

            vocab.insert(word.to_string(), VocabEntry { units, freq, uset });
        } else {
            eprintln!("Invalid line format: {}", line);
        }
    }

    println!("fn [read_vocab] used {:?}", fn_start.elapsed());

    Ok(vocab)
}


/*
fn get_stats(vocab: &HashMap<String, u32>) -> HashMap<(String, String), u32> {
    let mut pairs = HashMap::new();
    for (word, freq) in vocab.iter() {
        let symbols: Vec<&str> = word.split_whitespace().collect();
        for window in symbols.windows(2) {
            let pair = (window[0].to_string(), window[1].to_string());
            *pairs.entry(pair).or_insert(0) += freq;
        }
    }
    pairs
}
 */

fn get_pairs_stats_naive(vocab: &HashMap<String, VocabEntry>) -> HashMap<(String, String), i32> {
    let fn_start = Instant::now();
    let mut pairs_stats = HashMap::new();

    for entry in vocab.values() {
        for window in entry.units.windows(2) {
            if window.len() == 2 {
                let pair = (window[0].clone(), window[1].clone());
                // Increase the frequency of each pair by 1 for every occurrence.
                *pairs_stats.entry(pair).or_insert(0) += entry.freq;
            }
        }
        break;
    }

    println!("fn [get_pairs_stats naive] used {:?}", fn_start.elapsed());
    pairs_stats
}

fn get_pairs_stats_parallel(vocab: &HashMap<String, VocabEntry>) -> HashMap<(String, String), i32> {
    let pairs_stats = Mutex::new(HashMap::new());

    vocab.par_iter().for_each(|(_key, entry)| {
        let mut local_map = HashMap::new();
        for window in entry.units.windows(2) {
            if window.len() == 2 {
                let pair = (window[0].clone(), window[1].clone());
                *local_map.entry(pair).or_insert(0) += entry.freq;
            }
        }

        let start = Instant::now();
        let mut global_map = pairs_stats.lock().unwrap();
        for (pair, freq) in local_map {
            *global_map.entry(pair).or_insert(0) += freq;
        }
        println!("combine into global map {:?}", start.elapsed());
    });

    let start = Instant::now();
    let locked_map = pairs_stats.into_inner().unwrap();
    println!("into_inner().unwrap() {:?}", start.elapsed());

    locked_map
}


fn get_pairs_stats_chunk(vocab: &HashMap<String, VocabEntry>, n_jobs: isize) -> HashMap<(String, String), i32> {
    let fn_start = Instant::now();
    // Adjust n_jobs based on the available cores if n_jobs is -1
    let n_jobs = if n_jobs == -1 { num_cpus::get() } else { n_jobs as usize };

    // Build a custom thread pool with n_jobs threads
    let pool = ThreadPoolBuilder::new().num_threads(n_jobs).build().unwrap();

    let start = Instant::now();
    let vocab_values: Vec<&VocabEntry> = vocab.values().collect();
    println!("get vector of vocab values {:?}", start.elapsed());

    let chunk_size = (vocab_values.len()-1)/n_jobs + 1;
    println!("n_jobs: {:?}, len: {:?}, chunk_size: {:?}", n_jobs, vocab_values.len(), chunk_size);

    // Create partitions and process each partition in parallel
    let start = Instant::now();
    let pairs_stats: Vec<HashMap<(String, String), i32>> = vocab_values
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut local_pairs_stats = HashMap::new();
            {
                let start = Instant::now();
                for entry in chunk {
                    for window in entry.units.windows(2) {
                        if window.len() == 2 {
                            let pair = (window[0].clone(), window[1].clone());
                            *local_pairs_stats.entry(pair).or_insert(0) += entry.freq;
                        }
                    }
                }
                println!("process chunk {:?}", start.elapsed());
            }
            local_pairs_stats
        })
        .collect();
    println!("collect all chunk {:?}", start.elapsed());

    // Combine pairs_stats from all partitions
    let mut total_pairs_stats = HashMap::new();
    let start = Instant::now();
    for map in pairs_stats {
        for (pair, freq) in map {
            *total_pairs_stats.entry(pair).or_insert(0) += freq;
        }
    }
    println!("combine map {:?}", start.elapsed());

    println!("fn [get_pairs_stats_chunk] used {:?}", fn_start.elapsed());
    total_pairs_stats
}

/*
fn merge_vocab(pair: &(String, String), vocab: &HashMap<String, u32>) -> HashMap<String, u32> {
    let mut new_vocab = HashMap::new();
    let bigram = pair.0.clone() + " " + &pair.1;
    let replacement = pair.0.clone() + &pair.1;
    for (word, freq) in vocab {
        let new_word = word.replace(&bigram, &replacement);
        new_vocab.insert(new_word, *freq);
    }
    new_vocab
}

fn bpe(mut vocab: HashMap<String, u32>, num_merges: u32) -> HashSet<String> {
    for _ in 0..num_merges {
        let pairs = get_stats(&vocab);
        if let Some(pair) = pairs.iter().max_by_key(|entry| entry.1) {
            vocab = merge_vocab(&pair.0, &vocab);
        }
    }
    let mut tokens = HashSet::new();
    for word in vocab.keys() {
        word.split_whitespace().for_each(|token| { tokens.insert(token.to_string()); });
    }
    tokens
}
*/

/*
fn main() -> io::Result<()> {
    let path = "path/to/your/vocab/file.txt"; // Update this path to your vocabulary file
    let vocab = get_vocab(path)?;

    let num_merges = 10;
    let tokens = bpe(vocab, num_merges);

    println!("Tokens: {:?}", tokens);
    Ok(())
}
 */

fn get_vocab_path() -> PathBuf {
    let vocab_file = "proj/lifematrix/TransformerLM/data/generated/WMT-14/vocab.en";

    // Obtain the home directory as a PathBuf
    if let Some(home_dir) = dirs::home_dir() {
        // Concatenate the home directory path with the relative file path
        let absolute_path = home_dir.join(vocab_file);

        // Convert the PathBuf to a String, if needed
        match absolute_path.to_str() {
            Some(path_str) => println!("The absolute path is: {}", path_str),
            None => panic!("Could not convert the path to a string."),
        }
        absolute_path
    } else {
        panic!("Could not find your home directory!");
    }
}

fn main() -> io::Result<()> {
    // ThreadPoolBuilder::new().num_threads(n_jobs as usize) // Set the desired number of threads
    //    .build_global()
    //    .unwrap();

    let vocab_path = get_vocab_path();
    println!("vocab path is: {:?}", vocab_path);

    let start = Instant::now();
    let vocab = read_vocab(vocab_path, -1)?;
    println!("Execution time of read_vocab: {:?}", start.elapsed());

    let start_fn = Instant::now();
    let pairs_stats = get_pairs_stats_chunk(&vocab, -1);
    println!("Execution time of get_pairs_stats_chunk: {:?}", start_fn.elapsed());

    if let Some((best_pair, best_pair_freq)) = pairs_stats.iter().max_by_key(|entry| entry.1) {
        println!("The pair with the maximum frequency is ({:?}) with frequency {}", best_pair, best_pair_freq);
    } else {
        println!("No pairs found.");
    }
    //let num_merges = 10;
    // let tokens = bpe(vocab?, num_merges);

    // println!("Tokens: {:?}", tokens);
    Ok(())
}

