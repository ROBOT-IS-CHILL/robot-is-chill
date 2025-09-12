use std::{alloc::{GlobalAlloc, Layout, System}, any::Any, backtrace::Backtrace, cell::Cell, error::Error, io::{stdin, Read}, process::ExitCode, sync::{atomic::{AtomicBool, AtomicUsize, Ordering::*}, Arc}};
use macrosia::{Executor, TextMacro, VariableRegistry};

thread_local! {
    static BACKTRACE: Cell<Option<Backtrace>> = const { Cell::new(None) };
}

struct LimitAlloc;
static PANICKING: AtomicBool = AtomicBool::new(false);
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static MEMORY_LIMIT: usize = 2 * 1024 * 1024;
unsafe impl GlobalAlloc for LimitAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let panicking = PANICKING.load(Relaxed);
        let mem_left = ALLOCATED.load(Relaxed);
        if !panicking && mem_left.checked_add(layout.size()).is_none_or(|v| v > MEMORY_LIMIT) {
            panic!("ran out of memory")
        }
        let ret = unsafe { System.alloc(layout) };
        if !ret.is_null() {
            ALLOCATED.fetch_add(layout.size(), Relaxed);
        }
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
        ALLOCATED.fetch_sub(layout.size(), Relaxed);
    }
}

#[global_allocator]
static GLOBAL: LimitAlloc = LimitAlloc;

fn main() -> ExitCode {
    std::panic::set_hook(Box::new(|_| {
        PANICKING.store(true, Relaxed);
        let trace = Backtrace::capture();
        BACKTRACE.with(move |b| b.set(Some(trace)));
    }));
    let res = std::panic::catch_unwind(exec).unwrap_or_else(|e| {
        let string: String = match e.downcast::<String>() {
            Ok(str) => *str,
            Err(e) => match e.downcast::<&str>() {
                Ok(str) => String::from(*str),
                Err(_) => "<payload was not a string>".into()
            }
        };
        Err(Box::new(ErrorBt(string)))
    });
    match res {
        Ok(_) => ExitCode::SUCCESS,
        Err(err) => {
            print!("{err}");
            ExitCode::FAILURE
        }
    }
}

struct ErrorBt(String);
impl Error for ErrorBt {}
impl std::fmt::Display for ErrorBt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "panic: {}\n{}", self.0, BACKTRACE.with(|b| b.take()).ok_or(std::fmt::Error)? )}
}
impl std::fmt::Debug for ErrorBt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "panic: {}\n{:?}", self.0, BACKTRACE.with(|b| b.take()).ok_or(std::fmt::Error)? )}
}

struct StepLimitError(u32);
impl Error for StepLimitError {}
impl std::fmt::Display for StepLimitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "reached step limit of {}", self. )}
}
impl std::fmt::Debug for ErrorBt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "panic: {}\n{:?}", self.0, BACKTRACE.with(|b| b.take()).ok_or(std::fmt::Error)? )}
}



fn exec() -> Result<Cow<'static, u8>, Box<dyn Error>> {
    let mut exec;
    let step_limit;
    let mut program = vec![];
    {
        let mut stdin = stdin().lock();
        let mut buf = [0u8; 4];
        let context = stdin.read_exact(&mut buf[0..=0])?;
        stdin.read_exact(&mut buf)?;
        step_limit = u32::from_le_bytes(buf);
        exec = Executor::new(buf[0]);
        // Format: <context><step limit>\0<name_len: le u32><name><value_len: le u32><value>\0...\1<exec>
        loop {
            stdin.read_exact(&mut buf[0..=0])?;
            if buf[0] != b'\0' { break; }
            stdin.read_exact(&mut buf)?;
            let name_len = u32::from_le_bytes(buf);
            let mut name_buf = vec![0; name_len as usize];
            stdin.read_exact(&mut name_buf)?;
            let value_len = u32::from_le_bytes(buf);
            let mut value_buf = vec![0; value_len as usize];
            stdin.read_exact(&mut value_buf)?;
            exec.add_macro(TextMacro {
                name: Arc::new(name_buf),
                source: Arc::new(value_buf),
                description: Arc::new(String::new())
            });
        }
        exec = exec.with_stdlib();
        stdin.read_to_end(&mut program)?;
    }
    let mut var_reg = VariableRegistry::new();
    let genr = exec.evaluate(&program, &mut var_reg, Some(step_limit as usize));
    let mut steps = 0;
    loop {
        let Some(res) = genr() else {
            steps += 1;
            continue;
        };
        break res;
    }
}
