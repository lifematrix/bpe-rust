
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
use std::hash::Hash;
use num_cpus;
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::{Arc, Mutex};
use std::cmp::Ordering;


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

fn merge_and_stats(vocab: &mut HashMap<String, VocabEntry>, 
                   pairs_stats: &Arc<Mutex<HashMap<(String, String), i32>>>, 
                   best_pair: &(String, String),
                   n_jobs: isize) {

    let new_unit = best_pair.0.clone() + &best_pair.1;

    let n_jobs = if n_jobs == -1 { num_cpus::get() } else { n_jobs as usize };
    let pool = ThreadPoolBuilder::new().num_threads(20) // Set the desired number of threads
        .build()
        .unwrap();

    pool.install(|| {
        vocab.par_iter_mut().for_each(|(_word, entry)| {
            let mut i = 0;

            while i < entry.units.len()-1 {
                if &entry.units[i] == &best_pair.0 && &entry.units[i+1] == &best_pair.1 {
                    entry.units[i] = new_unit.clone();
                    entry.units.remove(i+1);

                    { 
                        if i>= 1 {
                            let mut ps = pairs_stats.lock().unwrap();
                            let k1 = (entry.units[i-1].clone(), best_pair.0.clone());
                            // println!("derease key {:?} from stats", k1);
                            *ps.entry(k1.clone()).or_insert(0) -= entry.freq;
                            if *ps.entry(k1.clone()).or_insert(0) == 0 {
                                ps.remove(&k1);
                            }
                            let k2 = (entry.units[i-1].clone(), new_unit.clone());
                            *ps.entry(k2).or_insert(0) += entry.freq;
                        }
                        if i < entry.units.len() - 1 {
                            let mut ps = pairs_stats.lock().unwrap();
                            let k1 = (best_pair.1.clone(), entry.units[i+1].clone());
                            *ps.entry(k1.clone()).or_insert(0) -= entry.freq;
                            // println!("derease key {:?} from stats", k1);
                            if *ps.entry(k1.clone()).or_insert(0) == 0 {
                                ps.remove(&k1);
                            }
                            let k2 = (new_unit.clone(), entry.units[i+1].clone());
                            *ps.entry(k2).or_insert(0) += entry.freq;
                        }
                    }
                }
                i += 1;
            }
        });
    });
    let mut ps = pairs_stats.lock().unwrap();
    ps.remove(best_pair);
}

fn merge_vocab_parallel(vocab: &mut HashMap<String, VocabEntry>, best_pair: &(String, String), n_jobs: isize) {
    let new_unit = best_pair.0.clone() + &best_pair.1;

    let n_jobs = if n_jobs == -1 { num_cpus::get() } else { n_jobs as usize };
    let pool = ThreadPoolBuilder::new().num_threads(n_jobs) // Set the desired number of threads
        .build()
        .unwrap();

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


/*
            let Some((local_best_pair, local_best_pair_freq)) = cur_pairs_stats.iter().max_by(|x, y| {
                    x.1.cmp(&y.1) // Compare values in descending order
                    .then_with(|| (x.0 .0.len() + x.0 .1.len()).cmp(&(y.0 .0.len() + y.0 .1.len()))) // Then compare key lengths
                    .then_with(|| x.0 .0.cmp(&y.0 .0).reverse()) // Then compare the first part of the key in ascending alphabetical order
                    .then_with(|| x.0 .1.cmp(&y.0 .1).reverse()) // Finally, compare the second part of the key in ascending alphabetical order if needed
 */                                                                //

fn cmp_pair(a: (&(String, String), &i32), b: (&(String, String), &i32)) -> Ordering {
    a.1.cmp(&b.1) // Compare values in descending order
        .then_with(|| (a.0 .0.len() + b.0 .1.len()).cmp(&(a.0 .0.len() + b.0 .1.len()))) // Then compare key lengths
        .then_with(|| a.0 .0.cmp(&b.0 .0).reverse()) // Then compare the first part of the key in ascending alphabetical order
        .then_with(|| a.0 .1.cmp(&b.0 .1).reverse()) // Finally, compare the second part of the key in ascending alphabetical order if needed
}

fn save_learned(vocab: &HashMap<String, VocabEntry>, 
                learned_tokens: &HashMap<String, i32>, 
                learned_pairs: &HashMap<(String, String), i32>,
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

    // Sort learned_pairs and write its length and elements to the file
    let mut sorted_pairs: Vec<_> = learned_pairs.iter().collect();
    sorted_pairs.sort_by(|a, b| cmp_pair(*a, *b).reverse());
    writeln!(file, "{}", sorted_pairs.len())?;
    for (key, &value) in sorted_pairs {
        writeln!(file, "{} {} {}", key.0, key.1, value)?;
    }

    Ok(())
}

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

    let start = Instant::now();
    let pairs_stats = get_pairs_stats_chunk(&vocab, -1);
    debug!("Execution time of get_pairs_stats_chunk: {:?}", start.elapsed());
    debug!("the initial length of pairs_stats: {}", pairs_stats.len());

    //let start = Instant::now();
    //let mux_pairs_stats = Arc::new(Mutex::new(pairs_stats));
    //debug!("Execution time of create pairs_stats_mux: {:?}", start.elapsed());

    // let mut cur_pairs_stats = pairs_stats.clone();
    // let mut mux_pairs_stats = Arc::new(Mutex::new(HashMap::<(String, String), i32>::new())); 
    let mut best_pair: (String, String);
    let mut best_pair_freq: i32;
    let mut cur_len: i32;
    let mux_pairs_stats = Arc::new(Mutex::new(pairs_stats));
    let mut learned_pairs: HashMap<(String, String), i32> = HashMap::new();

    let loop_start = Instant::now();
    for i in 0..n_rounds {
        // bar.inc(1);
        debug!("{}", "-".repeat(80));
        let start = Instant::now();
        {
            let cur_pairs_stats = mux_pairs_stats.lock().unwrap();
            cur_len = cur_pairs_stats.len() as i32;
            debug!("#{} the length of cur_pairs_stats before merge: {}", i, cur_pairs_stats.len());

            let start = Instant::now();
            // let Some((local_best_pair, local_best_pair_freq)) = cur_pairs_stats.iter().max_by_key(|entry| entry.1) else { todo!() };
            /*
            let Some((local_best_pair, local_best_pair_freq)) = cur_pairs_stats.iter().max_by(|x, y| {
                    x.1.cmp(&y.1) // Compare values in descending order
                    .then_with(|| (x.0 .0.len() + x.0 .1.len()).cmp(&(y.0 .0.len() + y.0 .1.len()))) // Then compare key lengths
                    .then_with(|| x.0 .0.cmp(&y.0 .0).reverse()) // Then compare the first part of the key in ascending alphabetical order
                    .then_with(|| x.0 .1.cmp(&y.0 .1).reverse()) // Finally, compare the second part of the key in ascending alphabetical order if needed
            }) else { todo!{} };
             */
            let Some((local_best_pair, local_best_pair_freq)) = cur_pairs_stats.iter().max_by(|a, b| cmp_pair(*a, *b)) else { todo!{} };

            debug!("#{} Execution of time find max pair: {:?}", i, start.elapsed());

            best_pair = local_best_pair.clone();
            best_pair_freq = local_best_pair_freq.clone();
        }
        debug!("#{} The pair with the maximum frequency is ({:?}) with frequency {}", i, best_pair, best_pair_freq);
    
        let start = Instant::now();
        merge_and_stats(&mut vocab, &mux_pairs_stats, &best_pair, -1);
        let cur_pairs_stats = mux_pairs_stats.lock().unwrap();
        let delta_len = (cur_pairs_stats.len() as i32) - cur_len;
        let elapsed = start.elapsed();
        debug!("#{} the length of cur_pairs_stats after merge: {}", i, cur_pairs_stats.len());
        debug!("#{} Execution time of merge_and_stats: {:?}, increase: {:?}, speed: {:.4}ms/pair", i, elapsed, delta_len, (elapsed.as_millis() as f64)/(delta_len as f64));

        *learned_tokens.entry(best_pair.0.clone() + &best_pair.1).or_insert(0) = best_pair_freq.clone();
        *learned_pairs.entry(best_pair.clone()).or_insert(0) = best_pair_freq.clone();

        let loop_elapsed = loop_start.elapsed();
        let speed = (loop_elapsed.as_millis() as f64)/((i+1) as f64);
        debug!("#{} Time used up to now: {:?}, {:.4}ms/iter, {:.4}iter(s)/sec", i, loop_elapsed, speed, 1000.0/speed); 
    }

    let start = Instant::now();
    save_learned(&vocab, &learned_tokens, &learned_pairs, learned_path);
    info!("\nExecution time of save_learned model to {:?}: {:?}\n", learned_path, start.elapsed());

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

