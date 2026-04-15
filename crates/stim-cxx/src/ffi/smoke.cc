#include "stim-cxx/include/stim_rs_bridge.h"

#include <sstream>
#include <string>

namespace stimrs::bridge {
namespace {
const char *platform_name() {
#if defined(_WIN32)
  return "windows";
#elif defined(__APPLE__)
  return "macos";
#elif defined(__linux__)
  return "linux";
#else
  return "unknown";
#endif
}

const char *vendor_mode() {
#if defined(STIM_RS_VENDOR_STIM_PRESENT)
  return "present";
#else
  return "absent";
#endif
}
}  // namespace

rust::String SmokeProbe::describe() const {
  std::ostringstream report;
  report << "stim-cxx bridge smoke"
         << " target=" << platform_name()
         << " vendor_stim=" << vendor_mode();
  return rust::String(report.str());
}

std::uint64_t SmokeProbe::weighted_checksum(rust::Slice<const std::uint64_t> values) const {
  std::uint64_t total = 0;
  std::size_t index = 0;
  for (std::uint64_t value : values) {
    total += static_cast<std::uint64_t>(++index) * value;
  }
  return total;
}

std::unique_ptr<SmokeProbe> new_smoke_probe() {
  return std::make_unique<SmokeProbe>();
}

}  // namespace stimrs::bridge
