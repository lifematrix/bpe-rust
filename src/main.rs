
use colored::*;
use fern;
use chrono;
use log::{info, warn, debug};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use dirs;
use std::time::Instant;
use std::sync::Mutex;
use std::hash::Hash;
use num_cpus;
use indicatif::{ProgressBar, ProgressStyle};


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
    // Adjust n_jobs based on the available cores if n_jobs is -1
    let n_jobs = if n_jobs == -1 { num_cpus::get() } else { n_jobs as usize };


    let pool = ThreadPoolBuilder::new().num_threads(n_jobs as usize) // Set the desired number of threads
        .build()
        .unwrap();

    let vocab_values: Vec<&VocabEntry> = vocab.values().collect();

    let chunk_size = (vocab_values.len()-1)/n_jobs + 1;
    //println!("n_jobs: {:?}, len: {:?}, chunk_size: {:?}", n_jobs, vocab_values.len(), chunk_size);

    let pairs_stats = pool.install(|| {
        vocab_values
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut local_pairs_stats = HashMap::new();
                for entry in chunk {
                    for window in entry.units.windows(2) {
                        if window.len() == 2 {
                            let pair = (window[0].clone(), window[1].clone());
                            *local_pairs_stats.entry(pair).or_insert(0) += entry.freq;
                        }
                    }
                }
                local_pairs_stats
            })
            .reduce_with(|mut map1, map2| {
                for (pair, freq) in map2 {
                    *map1.entry(pair).or_insert(0) += freq;
                }
                map1
            }).unwrap_or_else(HashMap::new)
    });

    pairs_stats
}

fn init_alphabet(vocab: &HashMap<String, VocabEntry>) -> HashMap<String, i32> {
    let mut alphabet_freq: HashMap<String, i32> = HashMap::new();

    for entry in vocab.values() {
        for unit in &entry.units {
            *alphabet_freq.entry(unit.clone()).or_insert(0) += entry.freq;
        }
    }

    alphabet_freq
}

fn merge_vocab(vocab: &mut HashMap<String, VocabEntry>, best_pair: &(String, String)) {
    let new_unit = best_pair.0.clone() + &best_pair.1; // Join the best_pair into a new String

    for entry in vocab.values_mut() {
        let mut idx_to_remove = vec![];
        for window in entry.units.windows(2).enumerate() {
            let (idx, pair) = window;
            if &pair[0] == &best_pair.0 && &pair[1] == &best_pair.1 {
                idx_to_remove.push(idx);
            }
        }

        for &idx in idx_to_remove.iter().rev() {
            entry.units.remove(idx); // Remove the element at idx
            entry.units.remove(idx); // idx now points to the next element, which was originally at idx+1
            entry.units.insert(idx, new_unit.clone()); // Insert new_unit at position idx
        }
    }
}

fn merge_vocab_parallel(vocab: &mut HashMap<String, VocabEntry>, best_pair: &(String, String), n_jobs: isize) {
    let new_unit = best_pair.0.clone() + &best_pair.1;

    let n_jobs = if n_jobs == -1 { num_cpus::get() } else { n_jobs as usize };
    let pool = ThreadPoolBuilder::new().num_threads(n_jobs) // Set the desired number of threads
        .build()
        .unwrap();

    /*
    pool.install(|| {
        vocab.par_iter_mut().for_each(|(_word, entry)| {
            let mut idx_to_remove = vec![];
            for (idx, window) in entry.units.windows(2).enumerate() {
                if window.len() == 2 && &window[0] == &best_pair.0 && &window[1] == &best_pair.1 {
                    idx_to_remove.push(idx);
                }
            }

            for &idx in idx_to_remove.iter().rev() {
                entry.units.remove(idx); // Remove the element at idx
                // Corrected to prevent panic from removing a non-existent element
                if idx < entry.units.len() {
                    entry.units.remove(idx); // Note: This second remove may not be needed; see explanation below.
                }
                entry.units.insert(idx, new_unit.clone()); // Insert new_unit at position idx
            }
        });
    });
     */
    pool.install(|| {
        vocab.par_iter_mut().for_each(|(_word, entry)| {
            let mut i = 0;

            while i < entry.units.len()-1 {
                if &entry.units[i] == &best_pair.0 && &entry.units[i+1] == &best_pair.1 {
                    entry.units[i] = new_unit.clone();
                    entry.units.remove(i+1);
                }
                i += 1;
            }
            

        });
    });
}

fn save_learned( vocab: &HashMap<String, VocabEntry>, learned_tokens: &HashMap<String, i32>, 
            learned_path: &String) -> io::Result<()> {
    let path = Path::new(learned_path);
    let mut file = File::create(path)?;

    // Sort vocab and write its length and elements to the file
    let mut sorted_vocab: Vec<_> = vocab.iter().collect();
    sorted_vocab.sort_by_key(|&(k, _)| k);
    writeln!(file, "{}", sorted_vocab.len())?;
    for (key, entry) in sorted_vocab {
        let units_str = entry.units.join(" ");
        writeln!(file, "{} {} {}", key, entry.freq, units_str)?;
    }

    // Sort learned_tokens and write its length and elements to the file
    let mut sorted_tokens: Vec<_> = learned_tokens.iter().collect();
    sorted_tokens.sort_by_key(|&(k, _)| k);
    writeln!(file, "{}", sorted_tokens.len())?;
    for (key, &value) in sorted_tokens {
        writeln!(file, "{} {}", key, value)?;
    }

    Ok(())
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

fn get_files_path() -> (String, String) {
    let vocab_file = "proj/lifematrix/TransformerLM/data/generated/WMT-14/vocab.en";
    let learned_file = "proj/lifematrix/TransformerLM/data/generated/WMT-14/bpe_learned_rust.en";

    // Obtain the home directory as a PathBuf
    if let Some(home_dir) = dirs::home_dir() {
        // Concatenate the home directory path with the relative file path
        let abs_vocab_path = home_dir.join(vocab_file);
        let abs_learned_path = home_dir.join(learned_file);

        // Convert the PathBuf to a String, if needed
        /*
        match absolute_path.to_str() {
            Some(path_str) => println!("The absolute path is: {}", path_str),
            None => panic!("Could not convert the path to a string."),
        }
        */
        (abs_vocab_path.to_str().unwrap().to_string(), abs_learned_path.to_str().unwrap().to_string())
    } else {
        panic!("Could not find your home directory!");
    }
}

fn bpe_learn(vocab_path: &String, learned_path: &String, max_size: i32) -> io::Result<()> {

    //let vocab_path = get_vocab_path();
    //println!("vocab path is: {:?}", vocab_path);

    let start = Instant::now();
    let mut vocab = read_vocab(vocab_path, -1)?;
    info!("Execution time of read_vocab: {:?}", start.elapsed());

    let start = Instant::now();
    let mut learned_tokens = init_alphabet(&vocab);
    println!("Execution time of init_alphabet: {:?}", start.elapsed());

    let n_rounds = max_size - learned_tokens.len() as i32;
    info!("n_rounds: {}, max_size: {}, # of initial basic tokens: {}", n_rounds, max_size, learned_tokens.len());

    let bar = ProgressBar::new(n_rounds as u64);
    bar.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} ({per_sec}, {eta_precise})")
        .progress_chars("#>-"));

    for i in 0..n_rounds {
        bar.inc(1);
        let start = Instant::now();
        let pairs_stats = get_pairs_stats_chunk(&vocab, -1);
        // debug!("Execution time of get_pairs_stats_chunk: {:?}", start.elapsed());
    
        let Some((best_pair, best_pair_freq)) = pairs_stats.iter().max_by_key(|entry| entry.1) else { todo!() };
        // println!("\n#{} The pair with the maximum frequency is ({:?}) with frequency {}\n", i, best_pair, best_pair_freq);
    
        let start = Instant::now();
        merge_vocab_parallel(&mut vocab, best_pair, -1);
        // debug!("Execution time of merge_vocab: {:?}", start.elapsed());

        *learned_tokens.entry(best_pair.0.clone() + &best_pair.1).or_insert(0) = best_pair_freq.clone();
    }

    let start = Instant::now();
    save_learned(&vocab, &learned_tokens, learned_path);
    info!("\nExecution time of save_learned: {:?}\n", start.elapsed());
    Ok(())
}

fn setup_logging() -> Result<(), fern::InitError> {
    // Format the current date
    let current_date = chrono::Local::now().format("%Y-%m-%d").to_string();
    // Create the log filename with the current date
    let log_file_name = format!("log/log-{}.log", current_date);

    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "{} {}: {}",
                chrono::Local::now().format("[%Y-%m-%d %H:%M:%S%.3f]").to_string().cyan(),
                match record.level() {
                    log::Level::Error => "ERROR".red(),
                    log::Level::Warn => "WARN".yellow(),
                    log::Level::Info => "INFO".green(),
                    log::Level::Debug => "DEBUG".blue(),
                    log::Level::Trace => "TRACE".purple(),
                },
                message
            ))
        })
        .chain(io::stderr()) // Log to stderr
        .chain(fern::log_file(log_file_name)?) // Log to a file with the current date in its name
        .apply()?;

    Ok(())
}

fn main() {
    setup_logging().expect("Failed to initialize logging.");
    let (vocab_path, learned_path) = get_files_path();
    info!("vocab path is: {:?},  learned_path: {:?}", vocab_path, learned_path);
    bpe_learn(&vocab_path, &learned_path, 37000);
}

