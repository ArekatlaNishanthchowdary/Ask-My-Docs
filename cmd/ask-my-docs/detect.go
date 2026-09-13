package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

// TEI publishes one image per CUDA compute capability and refuses to start on
// the wrong one ("Runtime compute cap N is not compatible"). Picking the tag by
// hand means reading your GPU's capability off a table, which is exactly the
// kind of lookup that gets done once, wrongly, and then silently runs on CPU.
//
// Ordered high to low so the first tag whose capability the GPU meets wins.
var teiImages = []struct {
	cap   float64
	image string
	gpus  string
}{
	{9.0, "hopper-1.7", "H100, H200"},
	{8.9, "89-1.7", "RTX 40xx, L4, L40S"},
	{8.6, "86-1.7", "RTX 30xx, A10, A40"},
	{8.0, "1.7", "A100, A30"},
	{7.5, "turing-1.7", "RTX 20xx, T4"},
}

const cpuImage = "cpu-1.7"

// teiImageFor maps a CUDA compute capability to a TEI tag. Anything below 7.5
// predates the kernels TEI ships, and anything above the newest known tag is
// too new for this TEI release (Blackwell, sm_100/sm_120) — both fall back to
// CPU, which is slower but actually runs.
func teiImageFor(computeCap float64) (image string, exact bool) {
	if computeCap <= 0 {
		return cpuImage, false
	}
	if computeCap > teiImages[0].cap {
		return cpuImage, false
	}
	for _, t := range teiImages {
		if computeCap >= t.cap {
			return t.image, true
		}
	}
	return cpuImage, false
}

type gpuInfo struct {
	Name       string
	ComputeCap float64
}

// detectGPU asks the driver rather than inferring from a model name — "RTX
// 4060" and "RTX 4060 Ti" are the same capability, but "A10" and "A100" are
// not, and nobody remembers which.
func detectGPU() (gpuInfo, error) {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=name,compute_cap", "--format=csv,noheader").Output()
	if err != nil {
		return gpuInfo{}, fmt.Errorf("nvidia-smi not usable: %w", err)
	}
	line, _, _ := strings.Cut(strings.TrimSpace(string(out)), "\n")
	name, capStr, ok := strings.Cut(line, ",")
	if !ok {
		return gpuInfo{}, fmt.Errorf("unexpected nvidia-smi output: %q", line)
	}
	c, err := strconv.ParseFloat(strings.TrimSpace(capStr), 64)
	if err != nil {
		return gpuInfo{}, fmt.Errorf("unparsable compute capability %q: %w", capStr, err)
	}
	return gpuInfo{Name: strings.TrimSpace(name), ComputeCap: c}, nil
}

// dockerSeesGPU reports whether the container runtime can actually pass the GPU
// through. A working nvidia-smi on the host says nothing about this: without
// the NVIDIA Container Toolkit the reservation fails at `up` time with an error
// about device driver "nvidia", which reads like a Docker bug rather than a
// missing package.
func dockerSeesGPU() bool {
	out, err := exec.Command("docker", "info", "--format", "{{json .Runtimes}}").Output()
	if err != nil {
		return false
	}
	return strings.Contains(string(out), "nvidia")
}

// upsertEnv sets key=value in an .env file, replacing an existing assignment
// rather than appending a second one that the first would win over.
func upsertEnv(path, key, value string) error {
	raw, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	lines := []string{}
	if len(raw) > 0 {
		lines = strings.Split(strings.ReplaceAll(string(raw), "\r\n", "\n"), "\n")
	}
	found := false
	for i, l := range lines {
		k, _, ok := strings.Cut(strings.TrimPrefix(strings.TrimSpace(l), "export "), "=")
		if ok && strings.TrimSpace(k) == key {
			lines[i] = key + "=" + value
			found = true
			break
		}
	}
	if !found {
		// Strip every trailing blank, not just one: a file that already ends
		// in a blank line otherwise grows a gap per key appended.
		for len(lines) > 0 && strings.TrimSpace(lines[len(lines)-1]) == "" {
			lines = lines[:len(lines)-1]
		}
		lines = append(lines, key+"="+value, "")
	}
	return os.WriteFile(path, []byte(strings.Join(lines, "\n")), 0o644)
}

// cmdDetect picks the reranker image for this machine and writes it to .env.
// Offline and service-free, like `chunks` — it runs before anything is up.
func cmdDetect(args []string) error {
	fs := flag.NewFlagSet("detect", flag.ExitOnError)
	envPath := fs.String("env", ".env", "env file to update")
	dry := fs.Bool("n", false, "print what would change without writing")
	_ = fs.Parse(args)

	gpu, err := detectGPU()
	switch {
	case err != nil:
		fmt.Printf("GPU:      none detected (%v)\n", err)
	default:
		fmt.Printf("GPU:      %s (compute %.1f)\n", gpu.Name, gpu.ComputeCap)
	}

	image, exact := teiImageFor(gpu.ComputeCap)
	usable := exact && dockerSeesGPU()

	switch {
	case exact && !usable:
		// The most confusing failure this command exists to prevent: the right
		// GPU, the right tag, and a container that cannot see the device.
		fmt.Printf("Docker:   cannot pass through GPUs — the NVIDIA Container Toolkit is not installed\n")
		fmt.Printf("          https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html\n")
		fmt.Printf("          Falling back to CPU; re-run this after installing it.\n")
		image = cpuImage
	case !exact && gpu.ComputeCap > 0:
		fmt.Printf("Note:     compute %.1f has no TEI 1.7 image (supported: 7.5-9.0) — using CPU\n", gpu.ComputeCap)
	}

	compose := "docker-compose.yml"
	if usable {
		compose = "docker-compose.yml:docker-compose.gpu.yml"
	}

	fmt.Printf("Reranker: %s\n", image)
	fmt.Printf("Compose:  %s\n", strings.ReplaceAll(compose, ":", " + "))

	if *dry {
		fmt.Printf("\n(dry run — %s not written)\n", *envPath)
		return nil
	}
	for _, kv := range [][2]string{
		{"TEI_IMAGE", image},
		{"COMPOSE_FILE", compose},
		// Compose splits COMPOSE_FILE on ";" on Windows and ":" elsewhere.
		// Pinning the separator keeps one written .env valid on both.
		{"COMPOSE_PATH_SEPARATOR", ":"},
	} {
		if err := upsertEnv(*envPath, kv[0], kv[1]); err != nil {
			return fmt.Errorf("writing %s: %w", *envPath, err)
		}
	}
	fmt.Printf("\nWrote %s. Now run: docker compose up -d\n", *envPath)
	return nil
}
