#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

static llvm::cl::opt<std::string>
    DevicePath("device", llvm::cl::desc("Device CIR module (.mlir)"),
               llvm::cl::value_desc("file"), llvm::cl::Required);

static llvm::cl::opt<std::string>
    HostPath("host", llvm::cl::desc("Host CIR module (.mlir)"),
             llvm::cl::value_desc("file"), llvm::cl::Required);

static llvm::cl::opt<std::string>
    OutputPath("o", llvm::cl::desc("Output file (default: stdout)"),
               llvm::cl::value_desc("file"), llvm::cl::init(""));

static llvm::cl::opt<std::string>
    HostOut("host-out", llvm::cl::desc("Output host CIR (.mlir)"),
            llvm::cl::value_desc("file"), llvm::cl::Required);

static llvm::cl::opt<std::string>
    DeviceOut("device-out", llvm::cl::desc("Output device CIR (.mlir)"),
              llvm::cl::value_desc("file"), llvm::cl::Required);

static FailureOr<OwningOpRef<ModuleOp>> parseModule(StringRef path,
                                                    MLIRContext &ctx) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return failure();

  llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(std::move(*buffer), llvm::SMLoc());

  auto mod = parseSourceFile<ModuleOp>(sm, &ctx);
  if (!mod)
    return failure();

  return OwningOpRef<ModuleOp>(mod.release());
}

static LogicalResult writeModuleToFile(ModuleOp mod, llvm::StringRef outPath) {
  std::string errMsg;
  auto outFile = mlir::openOutputFile(outPath, &errMsg);
  if (!outFile) {
    llvm::errs() << "error: cannot open output " << outPath << ": " << errMsg
                 << "\n";
    return failure();
  }

  mod.print(outFile->os());
  outFile->os() << "\n";
  outFile->keep();
  return success();
}

static FailureOr<ModuleOp> findNestedModuleBySymbol(ModuleOp container,
                                                    llvm::StringRef symName) {
  // Since the nested modules are direct children, SymbolTable lookup is enough.
  Operation *sym = SymbolTable::lookupSymbolIn(container, symName);
  if (!sym)
    return failure();
  auto m = dyn_cast<ModuleOp>(sym);
  if (!m)
    return failure();
  return m;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "cir-bundle-host-device\n");

  mlir::DialectRegistry registry;
  registry
      .insert<mlir::BuiltinDialect, mlir::arith::ArithDialect, cir::CIRDialect,
              mlir::memref::MemRefDialect, mlir::LLVM::LLVMDialect,
              mlir::DLTIDialect, mlir::omp::OpenMPDialect>();

  mlir::MLIRContext ctx(registry);

  ctx.loadDialect<mlir::BuiltinDialect>();
  ctx.loadDialect<mlir::arith::ArithDialect>();
  ctx.loadDialect<cir::CIRDialect>();
  ctx.loadDialect<mlir::memref::MemRefDialect>();
  ctx.loadDialect<mlir::LLVM::LLVMDialect>();
  ctx.loadDialect<mlir::DLTIDialect>();
  ctx.loadDialect<mlir::omp::OpenMPDialect>();
  // If you want strict parsing, you can omit these two lines.
  // If you want to be robust when files contain extra dialects:
  ctx.allowUnregisteredDialects();

  // Makes sure the dialects are actually constructed/available:
  ctx.loadAllAvailableDialects();

  auto devOrErr = parseModule(DevicePath, ctx);
  if (failed(devOrErr)) {
    llvm::errs() << "error: failed to parse device module\n";
    return 1;
  }
  auto hostOrErr = parseModule(HostPath, ctx);
  if (failed(hostOrErr)) {
    llvm::errs() << "error: failed to parse host module\n";
    return 1;
  }

  OwningOpRef<ModuleOp> dev = std::move(*devOrErr);
  OwningOpRef<ModuleOp> host = std::move(*hostOrErr);

  // Create container.
  OwningOpRef<ModuleOp> container = ModuleOp::create(UnknownLoc::get(&ctx));
  OpBuilder b(&ctx);

  // Ensure nested modules have symbol names.
  (*host).getOperation()->setAttr(SymbolTable::getSymbolAttrName(),
                                  b.getStringAttr("host"));
  (*dev).getOperation()->setAttr(SymbolTable::getSymbolAttrName(),
                                 b.getStringAttr("device"));

  (*host).getOperation()->setAttr("cir.gpu.kind", b.getStringAttr("host"));
  (*dev).getOperation()->setAttr("cir.gpu.kind", b.getStringAttr("device"));

  // Move them under the container.
  container->getBody()->push_back(host.release());
  container->getBody()->push_back(dev.release());

  // Add bundle metadata at top-level.
  NamedAttrList bundle;
  bundle.append("host", FlatSymbolRefAttr::get(&ctx, "host"));
  bundle.append("device", FlatSymbolRefAttr::get(&ctx, "device"));
  (*container)
      .getOperation()
      ->setAttr("cir.gpu.bundle", b.getDictionaryAttr(bundle));
  (*container)
      .getOperation()
      ->setAttr("cir.gpu.kernel_map", b.getArrayAttr({}));

  // Re-find nested modules inside container.
  auto hostNestedOrErr = findNestedModuleBySymbol(*container, "host");
  auto devNestedOrErr = findNestedModuleBySymbol(*container, "device");
  if (failed(hostNestedOrErr) || failed(devNestedOrErr)) {
    llvm::errs()
        << "error: could not find nested host/device modules in container\n";
    return 1;
  }

  ModuleOp hostNested = *hostNestedOrErr;
  ModuleOp devNested = *devNestedOrErr;

  // Emit each as a standalone module:
  // clone() gives you an owning op you can print independently.
  OwningOpRef<ModuleOp> hostOutMod = cast<ModuleOp>(hostNested->clone());
  OwningOpRef<ModuleOp> devOutMod = cast<ModuleOp>(devNested->clone());

  if (failed(writeModuleToFile(*hostOutMod, HostOut)))
    return 1;
  if (failed(writeModuleToFile(*devOutMod, DeviceOut)))
    return 1;

  return 0;
}
