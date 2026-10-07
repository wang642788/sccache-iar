// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::{
    compiler::{
        CCompileCommand, Cacheable, ColorMode, CompileCommand, CompilerArguments, Language,
        SingleCompileCommand,
        args::{
            ArgDisposition, ArgInfo, ArgToStringResult, ArgsIter, Argument, FromArg, IntoArg,
            PathTransformerFn, SearchableArgInfo,
        },
        c::{ArtifactDescriptor, CCompilerImpl, CCompilerKind, ParsedArguments},
    },
    counted_array, dist,
    errors::*,
    mock_command::{CommandChild, CommandCreatorSync, RunCommand},
    util::run_input_output,
};
use async_trait::async_trait;
use log::Level::Trace;
use std::{
    collections::HashMap,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::{self, Stdio},
};

/// The IAR C/C++ compilers (`iccarm`, `iccrx`, `iccriscv`, ...).
#[derive(Clone, Debug)]
pub struct Iar {
    pub version: Option<String>,
}

#[async_trait]
impl CCompilerImpl for Iar {
    fn kind(&self) -> CCompilerKind {
        CCompilerKind::Iar
    }

    // C++ is selected with `--c++`, which is hashed with the other arguments.
    fn plusplus(&self) -> bool {
        false
    }

    fn version(&self) -> Option<String> {
        self.version.clone()
    }

    fn parse_arguments(
        &self,
        arguments: &[OsString],
        cwd: &Path,
        _env_vars: &[(OsString, OsString)],
    ) -> CompilerArguments<ParsedArguments> {
        parse_arguments(arguments, cwd, &ARGS[..])
    }

    async fn preprocess<T>(
        &self,
        creator: &T,
        executable: &Path,
        parsed_args: &ParsedArguments,
        cwd: &Path,
        env_vars: &[(OsString, OsString)],
        _may_dist: bool,
        _rewrite_includes_only: bool,
        _preprocessor_cache_mode: bool,
    ) -> Result<process::Output>
    where
        T: CommandCreatorSync,
    {
        preprocess(creator, executable, parsed_args, cwd, env_vars).await
    }

    fn generate_compile_commands<T>(
        &self,
        _path_transformer: &mut dist::PathTransformer,
        executable: &Path,
        parsed_args: &ParsedArguments,
        cwd: &Path,
        env_vars: &[(OsString, OsString)],
        _rewrite_includes_only: bool,
    ) -> Result<(
        Box<dyn CompileCommand<T>>,
        Option<dist::CompileCommand>,
        Cacheable,
    )>
    where
        T: CommandCreatorSync,
    {
        generate_compile_commands(executable, parsed_args, cwd, env_vars).map(
            |(command, dist_command, cacheable)| {
                (CCompileCommand::new(command), dist_command, cacheable)
            },
        )
    }
}

/// Whether `executable` is named like an IAR compiler, i.e. `icc<target>`.
pub fn is_iar_like(executable: &Path) -> bool {
    executable
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
        .is_some_and(|s| s.len() > 3 && s.starts_with("icc"))
}

/// Ask `executable` for its version, returning it only if it is an IAR compiler.
///
/// IAR compilers have no `-E`, so the generic detection can't identify them.
pub async fn detect_version<T>(
    creator: &T,
    executable: &Path,
    env: &[(OsString, OsString)],
) -> Option<String>
where
    T: CommandCreatorSync,
{
    let mut cmd = creator.clone().new_command_sync(executable);
    cmd.arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .envs(env.iter().map(|s| (&s.0, &s.1)));
    trace!("iar detect_version: {:?}", cmd);
    let output = cmd.spawn().await.ok()?.wait_with_output().await.ok()?;
    if !output.status.success() {
        return None;
    }
    parse_version(&String::from_utf8_lossy(&output.stdout))
}

// `IAR ANSI C/C++ Compiler V9.60.4.438/LNX for ARM`
fn parse_version(stdout: &str) -> Option<String> {
    let line = stdout.lines().map(str::trim).find(|l| !l.is_empty())?;
    (line.starts_with("IAR ") && line.contains("Compiler")).then(|| line.to_owned())
}

ArgData! { pub
    NotCompilationFlag,
    Output(PathBuf),
    PassThrough(OsString),
    TooHardFlag,
    TooHard(OsString),
}

use self::ArgData::*;

// Options written as `--option=value` need no entry: they are a single
// argument and are passed through as unknown flags. What must be listed is
// every option taking its value as a separate argument, so that the value is
// not mistaken for the input file.
counted_array!(pub static ARGS: [ArgInfo<ArgData>; _] = [
    take_arg!("--aapcs", OsString, Separated, PassThrough),
    take_arg!("--abi", OsString, Separated, PassThrough),
    take_arg!("--branch_protection", OsString, Separated, PassThrough),
    take_arg!("--cdecp", OsString, Separated, PassThrough),
    take_arg!("--cpu", OsString, Separated, PassThrough),
    take_arg!("--cpu_mode", OsString, Separated, PassThrough),
    take_arg!("--deprecated_feature_warnings", OsString, Separated, PassThrough),
    take_arg!("--diag_error", OsString, Separated, PassThrough),
    take_arg!("--diag_remark", OsString, Separated, PassThrough),
    take_arg!("--diag_suppress", OsString, Separated, PassThrough),
    take_arg!("--diag_warning", OsString, Separated, PassThrough),
    take_arg!("--diagnostics_format", OsString, Separated, PassThrough),
    take_arg!("--diagnostics_tables", OsString, Separated, TooHard),
    take_arg!("--dlib_config", OsString, Separated, PassThrough),
    take_arg!("--enable_hardware_workaround", OsString, Separated, PassThrough),
    take_arg!("--endian", OsString, Separated, PassThrough),
    take_arg!("--error_limit", OsString, Separated, PassThrough),
    take_arg!("--f", OsString, Separated, TooHard),
    take_arg!("--fpu", OsString, Separated, PassThrough),
    take_arg!("--lock_regs", OsString, Separated, PassThrough),
    take_arg!("--max_cost_constexpr_call", OsString, Separated, PassThrough),
    take_arg!("--max_depth_constexpr_call", OsString, Separated, PassThrough),
    flag!("--mfc", TooHardFlag),
    take_arg!("--output", PathBuf, CanBeConcatenated(b'='), Output),
    take_arg!("--pending_instantiations", OsString, Separated, PassThrough),
    take_arg!("--predef_macros", OsString, Concatenated(b'='), TooHard),
    take_arg!("--preinclude", OsString, Separated, PassThrough),
    take_arg!("--preprocess", OsString, Concatenated(b'='), TooHard),
    take_arg!("--public_equ", OsString, Separated, PassThrough),
    take_arg!("--runtime_checking", OsString, Separated, PassThrough),
    take_arg!("--section", OsString, Separated, PassThrough),
    take_arg!("--section_prefix", OsString, Separated, PassThrough),
    take_arg!("--source_encoding", OsString, Separated, PassThrough),
    take_arg!("--system_include_dir", OsString, Separated, PassThrough),
    take_arg!("--text_out", OsString, Separated, PassThrough),
    flag!("--version", NotCompilationFlag),
    take_arg!("--visibility", OsString, Separated, PassThrough),
    take_arg!("-D", OsString, CanBeSeparated, PassThrough),
    take_arg!("-I", OsString, CanBeSeparated, PassThrough),
    take_arg!("-f", OsString, Separated, TooHard),
    // List files: -l[c|C|D|E|a|A|b|B][N][H] file
    take_arg!("-l", OsString, Concatenated, TooHard),
    take_arg!("-o", PathBuf, CanBeSeparated, Output),
]);

/// Parse `arguments`, determining whether it is supported.
///
/// If any of the entries in `arguments` result in a compilation that
/// cannot be cached, return `CompilerArguments::CannotCache`.
/// If the commandline described by `arguments` is not compilation,
/// return `CompilerArguments::NotCompilation`.
/// Otherwise, return `CompilerArguments::Ok(ParsedArguments)`, with
/// the `ParsedArguments` struct containing information parsed from
/// `arguments`.
fn parse_arguments<S>(
    arguments: &[OsString],
    cwd: &Path,
    arg_info: S,
) -> CompilerArguments<ParsedArguments>
where
    S: SearchableArgInfo<ArgData>,
{
    // `--dependencies[=format] file` is both concatenated and separated, which
    // the argument table can't express, so pull it out first.
    let mut depfile = None;
    let mut remaining = Vec::with_capacity(arguments.len());
    let mut iter = arguments.iter();
    while let Some(arg) = iter.next() {
        let s = arg.to_string_lossy();
        if s == "--dependencies" || s.starts_with("--dependencies=") {
            match iter.next() {
                Some(path) => depfile = Some(PathBuf::from(path)),
                None => cannot_cache!("--dependencies"),
            }
        } else {
            remaining.push(arg.clone());
        }
    }

    let mut input_arg = None;
    let mut multiple_input = false;
    let mut output_arg = None;

    for arg in ArgsIter::new(remaining.into_iter(), arg_info) {
        let arg = try_or_cannot_cache!(arg, "argument parse");

        match arg.get_data() {
            Some(TooHardFlag) | Some(TooHard(_)) => {
                cannot_cache!(arg.flag_str().expect("Can't be Argument::Raw/UnknownFlag",))
            }
            Some(NotCompilationFlag) => return CompilerArguments::NotCompilation,
            Some(Output(p)) => output_arg = Some(p.clone()),
            Some(PassThrough(_)) => {}
            None => match arg {
                Argument::Raw(ref val) => {
                    if input_arg.is_some() {
                        multiple_input = true;
                    }
                    input_arg = Some(val.clone());
                }
                Argument::UnknownFlag(_) => {}
                _ => unreachable!(),
            },
        }
    }

    // Can't cache compilations with multiple inputs.
    if multiple_input {
        cannot_cache!("multiple input files");
    }
    let input = match input_arg {
        Some(i) => i,
        // We can't cache compilation without an input.
        None => cannot_cache!("no input file"),
    };
    let language = match Language::from_file_name(Path::new(&input)) {
        Some(l) => l,
        None => cannot_cache!("unknown source language"),
    };

    // Without -o the object file lands in the working directory.
    let output = match output_arg {
        Some(o) => o,
        None => match Path::new(&input).file_name() {
            Some(name) => Path::new(name).with_extension("o"),
            None => cannot_cache!("no output file"),
        },
    };
    // Both outputs can also name a directory (and --dependencies takes `+`),
    // in which case the compiler picks the file name.
    if cwd.join(&output).is_dir() {
        cannot_cache!("output is a directory");
    }
    if let Some(ref d) = depfile
        && (d.as_os_str() == "+" || cwd.join(d).is_dir())
    {
        cannot_cache!("--dependencies");
    }

    let mut outputs = HashMap::with_capacity(2);
    outputs.insert(
        "obj",
        ArtifactDescriptor {
            path: output,
            optional: false,
        },
    );
    if let Some(ref d) = depfile {
        outputs.insert(
            "d",
            ArtifactDescriptor {
                path: d.clone(),
                optional: false,
            },
        );
    }

    CompilerArguments::Ok(ParsedArguments {
        input: input.into(),
        double_dash_input: false,
        language,
        compilation_flag: OsString::new(),
        depfile,
        outputs,
        dependency_args: vec![],
        preprocessor_args: vec![],
        common_args: vec![],
        // The compiler records its whole command line in the object file
        // (`.comment`), so the object depends on every argument as written,
        // output paths and order included. Keep the original command line
        // here: it is hashed, and it is what both the preprocessor and the
        // compilation run.
        arch_args: arguments.to_vec(),
        unhashed_args: vec![],
        extra_dist_files: vec![],
        extra_hash_files: vec![],
        uses_external_assembler: false,
        msvc_show_includes: false,
        profile_generate: false,
        color_mode: ColorMode::Auto,
        suppress_rewrite_includes_only: false,
        too_hard_for_preprocessor_cache_mode: None,
    })
}

async fn preprocess<T>(
    creator: &T,
    executable: &Path,
    parsed_args: &ParsedArguments,
    cwd: &Path,
    env_vars: &[(OsString, OsString)],
) -> Result<process::Output>
where
    T: CommandCreatorSync,
{
    // The preprocessor output can only be written to a file, not to stdout.
    let tempdir = tempfile::Builder::new().prefix("sccache").tempdir()?;
    let preprocessed = tempdir.path().join("preprocessed.i");

    // With the whole command line, `--dependencies` writes the dependency
    // file here exactly as the compilation would. That is needed: on a cache
    // hit it is not restored from the cache, see `get_cached_or_compile`.
    let mut cmd = creator.clone().new_command_sync(executable);
    cmd.args(&parsed_args.arch_args)
        .arg("--preprocess=n")
        .arg(&preprocessed)
        .env_clear()
        .envs(env_vars.to_vec())
        .current_dir(cwd);

    if log_enabled!(Trace) {
        trace!("preprocess: {:?}", cmd);
    }

    let mut output = run_input_output(cmd, None).await?;
    output.stdout = fs::read(&preprocessed).context("failed to read preprocessor output")?;
    Ok(output)
}

fn generate_compile_commands(
    executable: &Path,
    parsed_args: &ParsedArguments,
    cwd: &Path,
    env_vars: &[(OsString, OsString)],
) -> Result<(
    SingleCompileCommand,
    Option<dist::CompileCommand>,
    Cacheable,
)> {
    trace!("compile");

    let command = SingleCompileCommand {
        executable: executable.to_owned(),
        // See `arch_args` in `parse_arguments`.
        arguments: parsed_args.arch_args.clone(),
        env_vars: env_vars.to_owned(),
        cwd: cwd.to_owned(),
        share_jobserver: false,
    };

    Ok((command, None, Cacheable::Yes))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::compiler::*;
    use crate::mock_command::*;
    use crate::test::utils::*;

    fn parse_arguments_(arguments: Vec<String>) -> CompilerArguments<ParsedArguments> {
        let args = arguments.iter().map(OsString::from).collect::<Vec<_>>();
        parse_arguments(&args, ".".as_ref(), &ARGS[..])
    }

    fn parsed(arguments: Vec<String>) -> ParsedArguments {
        match parse_arguments_(arguments) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {:?}", o),
        }
    }

    #[test]
    fn test_is_iar_like() {
        assert!(is_iar_like(Path::new("/opt/iar/arm/bin/iccarm")));
        assert!(is_iar_like(Path::new("iccrx.exe")));
        assert!(is_iar_like(Path::new("icc430")));
        // Intel's compiler.
        assert!(!is_iar_like(Path::new("/usr/bin/icc")));
        assert!(!is_iar_like(Path::new("/usr/bin/gcc")));
    }

    #[test]
    fn test_parse_version() {
        assert_eq!(
            Some("IAR ANSI C/C++ Compiler V9.60.4.438/LNX for ARM".to_owned()),
            parse_version("\n   IAR ANSI C/C++ Compiler V9.60.4.438/LNX for ARM\n")
        );
        assert_eq!(None, parse_version("icc (ICC) 2021.10.0 20230609"));
        assert_eq!(None, parse_version(""));
    }

    #[test]
    fn test_parse_arguments_simple() {
        let args = stringvec!["foo.c", "-o", "foo.o"];
        let ParsedArguments {
            input,
            language,
            outputs,
            depfile,
            preprocessor_args,
            common_args,
            arch_args,
            ..
        } = parsed(args.clone());
        assert_eq!(Some("foo.c"), input.to_str());
        assert_eq!(Language::C, language);
        assert_map_contains!(
            outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: PathBuf::from("foo.o"),
                    optional: false
                }
            )
        );
        assert_eq!(1, outputs.len());
        assert!(depfile.is_none());
        assert!(preprocessor_args.is_empty());
        assert!(common_args.is_empty());
        assert_eq!(ovec!["foo.c", "-o", "foo.o"], arch_args);
    }

    #[test]
    fn test_parse_arguments_default_output() {
        let ParsedArguments { outputs, .. } = parsed(stringvec!["src/foo.cpp", "--c++"]);
        assert_map_contains!(
            outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: PathBuf::from("foo.o"),
                    optional: false
                }
            )
        );
    }

    #[test]
    fn test_parse_arguments_output_long() {
        for args in [
            stringvec!["foo.c", "--output", "out/foo.o"],
            stringvec!["foo.c", "--output=out/foo.o"],
        ] {
            let ParsedArguments { outputs, .. } = parsed(args);
            assert_map_contains!(
                outputs,
                (
                    "obj",
                    ArtifactDescriptor {
                        path: PathBuf::from("out/foo.o"),
                        optional: false
                    }
                )
            );
        }
    }

    #[test]
    fn test_parse_arguments_cmake() {
        // The shape of a command line generated by CMake.
        let args = stringvec![
            "--c++",
            "--silent",
            "/src/foo.cpp",
            "-DFOO=1",
            "-D",
            "BAR",
            "-I/src/include",
            "--cpu",
            "Cortex-M55",
            "--fpu=VFPv5_D16",
            "--diag_suppress",
            "Pa205",
            "-e",
            "-Ohz",
            "--debug",
            "--dependencies=ns",
            "obj/foo.o.d",
            "-o",
            "obj/foo.o"
        ];
        let ParsedArguments {
            input,
            language,
            outputs,
            depfile,
            common_args,
            arch_args,
            ..
        } = parsed(args.clone());
        assert_eq!(Some("/src/foo.cpp"), input.to_str());
        assert_eq!(Language::Cxx, language);
        assert_map_contains!(
            outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: PathBuf::from("obj/foo.o"),
                    optional: false
                }
            ),
            (
                "d",
                ArtifactDescriptor {
                    path: PathBuf::from("obj/foo.o.d"),
                    optional: false
                }
            )
        );
        assert_eq!(Some(PathBuf::from("obj/foo.o.d")), depfile);
        assert!(common_args.is_empty());
        let expected: Vec<OsString> = args.iter().map(OsString::from).collect();
        assert_eq!(expected, arch_args);
    }

    #[test]
    fn test_parse_arguments_dependencies_without_format() {
        let ParsedArguments { depfile, .. } = parsed(stringvec![
            "foo.c",
            "--dependencies",
            "foo.d",
            "-o",
            "foo.o"
        ]);
        assert_eq!(Some(PathBuf::from("foo.d")), depfile);
    }

    #[test]
    fn test_parse_arguments_too_hard() {
        for args in [
            stringvec!["foo.c", "--preprocess=n", "foo.i"],
            stringvec!["foo.c", "--preprocess", "foo.i"],
            stringvec!["foo.c", "--predef_macros", "foo.txt"],
            stringvec!["foo.c", "-lC", "foo.lst", "-o", "foo.o"],
            stringvec!["foo.c", "-l", "foo.lst", "-o", "foo.o"],
            stringvec!["-f", "args.txt"],
            stringvec!["--f", "args.txt"],
            stringvec!["foo.c", "bar.c", "--mfc", "-o", "foo.o"],
            stringvec!["foo.c", "bar.c", "-o", "foo.o"],
            stringvec!["foo.c", "--dependencies=ns", "+", "-o", "foo.o"],
            stringvec!["foo.c", "--dependencies=ns"],
            stringvec!["-o", "foo.o"],
            stringvec!["foo.unknown", "-o", "foo.o"],
        ] {
            match parse_arguments_(args.clone()) {
                CompilerArguments::CannotCache(..) => {}
                o => panic!("Got unexpected parse result for {:?}: {:?}", args, o),
            }
        }
    }

    #[test]
    fn test_parse_arguments_not_compilation() {
        assert_eq!(
            CompilerArguments::NotCompilation,
            parse_arguments_(stringvec!["--version"])
        );
    }

    #[test]
    fn test_generate_compile_commands_verbatim() {
        let f = TestFixture::new();
        let args = stringvec![
            "--c++",
            "foo.cpp",
            "-DFOO",
            "--dependencies=ns",
            "foo.o.d",
            "-o",
            "foo.o"
        ];
        let parsed_args = parsed(args.clone());
        let (command, dist_command, cacheable) =
            generate_compile_commands(&f.bins[0], &parsed_args, f.tempdir.path(), &[]).unwrap();
        let expected: Vec<OsString> = args.iter().map(OsString::from).collect();
        assert_eq!(expected, command.arguments);
        assert!(dist_command.is_none());
        assert_eq!(Cacheable::Yes, cacheable);
    }

    #[test]
    fn test_detect_version() {
        let f = TestFixture::new();
        let creator = new_creator();
        let runtime = single_threaded_runtime();
        next_command(
            &creator,
            Ok(MockChild::new(
                exit_status(0),
                "IAR ANSI C/C++ Compiler V9.60.4.438/LNX for ARM\n",
                "",
            )),
        );
        let version = runtime.block_on(detect_version(&creator, &f.bins[0], &[]));
        assert_eq!(
            Some("IAR ANSI C/C++ Compiler V9.60.4.438/LNX for ARM".to_owned()),
            version
        );

        next_command(&creator, Ok(MockChild::new(exit_status(1), "", "nope")));
        assert_eq!(
            None,
            runtime.block_on(detect_version(&creator, &f.bins[0], &[]))
        );
    }
}
